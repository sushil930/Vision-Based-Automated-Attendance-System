"""Standard FastAPI web interface for face attendance and profile management."""

from __future__ import annotations

import base64
import logging
import socket
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from app import PROJECT_ROOT, recognize_group_image
from registration_app import (
    SUPPORTED_IMAGE_TYPES,
    delete_profile,
    detect_group_faces,
    edit_profile,
    profile_rows,
    save_assignments,
)


LOGGER = logging.getLogger(__name__)
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATE_DIR = PROJECT_ROOT / "templates"
UPLOAD_EXTENSIONS = set(SUPPORTED_IMAGE_TYPES)
REGISTRATION_SESSIONS: dict[str, list[dict[str, Any]]] = {}

app = FastAPI(title="Face Attendance")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Assignment(BaseModel):
    student_id: str = ""
    name: str = ""


class RegistrationSaveRequest(BaseModel):
    session_id: str
    assignments: list[Assignment]


class ProfileEditRequest(BaseModel):
    current_student_id: str
    new_student_id: str
    new_name: str


class ProfileDeleteRequest(BaseModel):
    student_id: str
    confirmed: bool = False


def image_to_data_url(image: np.ndarray) -> str:
    """Encode an RGB image for browser display."""
    with tempfile.SpooledTemporaryFile() as buffer:
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format="JPEG", quality=90)
        buffer.seek(0)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.read()).decode("ascii")


async def write_upload(upload: UploadFile) -> Path:
    """Validate an image upload and store it for one request only."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(400, "Upload a JPG, PNG, HEIC, or HEIF image.")
    contents = await upload.read()
    if not contents:
        raise HTTPException(400, "The uploaded image is empty.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
        temporary_file.write(contents)
        return Path(temporary_file.name)


def cleanup(path: Path) -> None:
    """Clean up the temporary user upload."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Could not remove temporary upload: %s", path)


def find_available_port(start: int = 8000, end: int = 8010) -> int:
    """Return the first available localhost port in the app's preferred range."""
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            if connection.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No available port found in the range {start}-{end}.")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(TEMPLATE_DIR / "index.html")


@app.get("/api/profiles")
def get_profiles() -> dict[str, Any]:
    return {"profiles": profile_rows()}


@app.post("/api/recognize")
async def recognize(image: UploadFile = File(...)) -> dict[str, Any]:
    upload_path = await write_upload(image)
    try:
        annotated, cards, rows, summary = recognize_group_image(str(upload_path))
    finally:
        cleanup(upload_path)
    return {
        "image": image_to_data_url(annotated) if annotated is not None else None,
        "faces": [{"image": image_to_data_url(crop), "label": label} for crop, label in cards],
        "rows": rows,
        "summary": summary,
    }


@app.post("/api/register/detect")
async def detect_unregistered(image: UploadFile = File(...)) -> dict[str, Any]:
    upload_path = await write_upload(image)
    try:
        annotated, cards, rows, detections, status = detect_group_faces(str(upload_path))
    finally:
        cleanup(upload_path)
    session_id = str(uuid4())
    REGISTRATION_SESSIONS[session_id] = detections
    return {
        "session_id": session_id,
        "image": image_to_data_url(annotated) if annotated is not None else None,
        "faces": [{"image": image_to_data_url(crop), "label": label} for crop, label in cards],
        "rows": rows,
        "status": status,
    }


@app.post("/api/register/save")
def save_registered_faces(request: RegistrationSaveRequest) -> dict[str, Any]:
    detections = REGISTRATION_SESSIONS.pop(request.session_id, None)
    if detections is None:
        raise HTTPException(400, "Detection session expired. Detect faces again.")
    if len(request.assignments) != len(detections):
        raise HTTPException(400, "Assignments do not match the detected faces.")
    rows = [[detection["label"], assignment.student_id, assignment.name] for detection, assignment in zip(detections, request.assignments)]
    profiles, status = save_assignments(detections, rows)
    return {"profiles": profiles, "status": status}


@app.post("/api/profiles/edit")
def update_profile(request: ProfileEditRequest) -> dict[str, Any]:
    profiles, status = edit_profile(request.current_student_id, request.new_student_id, request.new_name)
    return {"profiles": profiles, "status": status}


@app.post("/api/profiles/delete")
def remove_profile(request: ProfileDeleteRequest) -> dict[str, Any]:
    profiles, status = delete_profile(request.student_id, request.confirmed)
    return {"profiles": profiles, "status": status}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    port = find_available_port()
    LOGGER.info("Starting Face Attendance at http://127.0.0.1:%s", port)
    uvicorn.run(app, host="127.0.0.1", port=port)
