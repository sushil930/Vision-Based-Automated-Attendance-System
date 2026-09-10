"""FastAPI server exposing face recognition endpoints for the mobile app."""

from __future__ import annotations

import base64
import io
import json
import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from app import (
    GALLERY_FILE,
    PROJECT_ROOT,
    get_model,
    load_gallery,
    recognize_face,
    MATCH_THRESHOLD,
    MIN_MARGIN,
)

register_heif_opener()

LOGGER = logging.getLogger(__name__)
PROFILES_FILE = PROJECT_ROOT / "gallery" / "profiles.json"
CROPS_DIR = PROJECT_ROOT / "gallery" / "crops"

app = FastAPI(title="AI Attendance API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to an RGB numpy array."""
    image = Image.open(io.BytesIO(file_bytes))
    return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def crop_face(image_rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Create a padded RGB crop around one detected face."""
    x1, y1, x2, y2 = bbox.astype(int)
    px = int((x2 - x1) * 0.3)
    py = int((y2 - y1) * 0.4)
    return image_rgb[
        max(0, y1 - py): min(image_rgb.shape[0], y2 + py),
        max(0, x1 - px): min(image_rgb.shape[1], x2 + px),
    ]


def ndarray_to_b64jpg(arr: np.ndarray, quality: int = 80) -> str:
    """Encode an RGB numpy array as a base64 JPEG string."""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector."""
    normed = embedding.astype(np.float32)
    length = np.linalg.norm(normed)
    if length == 0:
        raise ValueError("Invalid zero-length embedding")
    return normed / length


def save_gallery(gallery: dict) -> None:
    """Atomically replace the gallery pickle file."""
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=GALLERY_FILE.parent, delete=False, suffix=".pkl"
    ) as tmp:
        pickle.dump(gallery, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(GALLERY_FILE)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/profiles")
def list_profiles():
    """Return all registered gallery profiles."""
    try:
        gallery = load_gallery()
    except FileNotFoundError:
        return {"profiles": []}
    profiles = [
        {
            "student_id": str(sid),
            "name": p["name"],
            "samples": len(p["embeddings"]),
        }
        for sid, p in sorted(gallery.items(), key=lambda x: str(x[0]))
    ]
    return {"profiles": profiles}


@app.post("/recognize")
async def recognize(image: UploadFile = File(...)):
    """Recognize faces in an uploaded group photo."""
    file_bytes = await image.read()
    try:
        image_rgb = decode_upload(file_bytes)
    except Exception as exc:
        raise HTTPException(400, f"Cannot decode image: {exc}")

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = get_model().get(image_bgr)
    gallery = load_gallery()

    results = []
    annotated = image_bgr.copy()

    for idx, face in enumerate(faces, 1):
        result = recognize_face(face.embedding, gallery)
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2 = min(x2, annotated.shape[1])
        y2 = min(y2, annotated.shape[0])

        colors = {"MATCH": (0, 180, 0), "REVIEW": (0, 180, 255), "UNKNOWN": (0, 0, 220)}
        color = colors[result["status"]]
        display_name = result["name"] if result["status"] != "UNKNOWN" else "Unknown"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated, f"{display_name} {result['score']:.2f}",
            (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )

        crop = crop_face(image_rgb, face.bbox)
        results.append({
            "index": idx,
            "name": display_name,
            "student_id": result["student_id"] if result["status"] != "UNKNOWN" else "-",
            "status": result["status"],
            "score": round(result["score"], 3),
            "crop_b64": ndarray_to_b64jpg(crop),
        })

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    counts = {"MATCH": 0, "REVIEW": 0, "UNKNOWN": 0}
    for r in results:
        counts[r["status"]] += 1

    return {
        "total_faces": len(faces),
        "counts": counts,
        "results": results,
        "annotated_b64": ndarray_to_b64jpg(annotated_rgb),
    }


@app.post("/review")
async def review(confirmed_indices: list[int] = Form(...)):
    """This is handled client-side — the mobile app reclassifies locally."""
    return {"status": "ok", "confirmed": confirmed_indices}


@app.post("/register")
async def register(
    image: UploadFile = File(...),
):
    """Detect unregistered faces in a group photo for registration."""
    file_bytes = await image.read()
    try:
        image_rgb = decode_upload(file_bytes)
    except Exception as exc:
        raise HTTPException(400, f"Cannot decode image: {exc}")

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = get_model().get(image_bgr)
    gallery = load_gallery()

    faces = sorted(faces, key=lambda f: (f.bbox[1], f.bbox[0]))
    unregistered = []

    for face in faces:
        match = recognize_face(face.embedding, gallery)
        if match["status"] == "MATCH":
            continue
        embedding = normalize_embedding(face.embedding)
        crop = crop_face(image_rgb, face.bbox)
        unregistered.append({
            "embedding": embedding.tolist(),
            "crop_b64": ndarray_to_b64jpg(crop),
        })

    return {
        "total_faces": len(faces),
        "registered_count": len(faces) - len(unregistered),
        "unregistered": unregistered,
    }


@app.post("/register/save")
async def register_save(data: dict):
    """Save registration assignments. Expects JSON: {assignments: [{embedding, student_id, name}, ...]}"""
    assignments = data.get("assignments", [])
    if not assignments:
        raise HTTPException(400, "No assignments provided")

    gallery = load_gallery()
    gallery_copy = {
        str(sid): {**p, "embeddings": list(p["embeddings"])}
        for sid, p in gallery.items()
    }

    saved = 0
    for a in assignments:
        sid = str(a["student_id"]).strip()
        name = a["name"].strip()
        if not sid or not name:
            continue
        embedding = np.asarray(a["embedding"], dtype=np.float32)

        if sid in gallery_copy:
            if gallery_copy[sid]["name"].casefold() != name.casefold():
                raise HTTPException(
                    400,
                    f"ID {sid} belongs to {gallery_copy[sid]['name']}. Use the existing name.",
                )
            gallery_copy[sid]["embeddings"].append(embedding)
        else:
            gallery_copy[sid] = {
                "student_id": sid,
                "name": name,
                "embeddings": [embedding],
            }
        saved += 1

    save_gallery(gallery_copy)
    return {"saved": saved}


@app.post("/add-samples")
async def add_samples(images: list[UploadFile] = File(...)):
    """Process images and return proposed sample additions."""
    gallery = load_gallery()
    model = get_model()
    proposals = []

    for upload in images:
        file_bytes = await upload.read()
        try:
            image_rgb = decode_upload(file_bytes)
        except Exception:
            continue
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        faces = model.get(image_bgr)

        for face in faces:
            embedding = face.embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding /= norm
            result = recognize_face(embedding, gallery)
            if result["score"] >= 0.55:
                crop = crop_face(image_rgb, face.bbox)
                proposals.append({
                    "student_id": result["student_id"],
                    "name": result["name"],
                    "score": round(result["score"], 3),
                    "auto_add": result["score"] >= 0.70,
                    "embedding": embedding.tolist(),
                    "crop_b64": ndarray_to_b64jpg(crop),
                })

    return {"proposals": proposals}


@app.post("/add-samples/commit")
async def add_samples_commit(data: dict):
    """Commit confirmed sample additions. Expects JSON: {samples: [{student_id, embedding}, ...]}"""
    samples = data.get("samples", [])
    if not samples:
        raise HTTPException(400, "No samples provided")

    gallery = load_gallery()
    added = 0
    for s in samples:
        sid = str(s["student_id"])
        if sid in gallery:
            embedding = np.asarray(s["embedding"], dtype=np.float32)
            gallery[sid]["embeddings"].append(embedding)
            added += 1

    if added > 0:
        save_gallery(gallery)
    return {"added": added}


@app.put("/profiles/{student_id}")
async def edit_profile(student_id: str, data: dict):
    """Edit a profile's student ID or name."""
    new_id = str(data.get("new_student_id", "")).strip()
    new_name = data.get("new_name", "").strip()
    if not new_id or not new_name:
        raise HTTPException(400, "Provide new_student_id and new_name")

    gallery = load_gallery()
    key = next((k for k in gallery if str(k) == student_id), None)
    if key is None:
        raise HTTPException(404, f"No profile with ID {student_id}")

    if new_id != student_id:
        existing = next((k for k in gallery if str(k) == new_id), None)
        if existing is not None:
            raise HTTPException(400, f"ID {new_id} already in use")

    gallery_copy = {
        str(sid): {**p, "embeddings": list(p["embeddings"])}
        for sid, p in gallery.items()
    }
    profile = gallery_copy.pop(str(key))
    profile["student_id"] = new_id
    profile["name"] = new_name
    gallery_copy[new_id] = profile
    save_gallery(gallery_copy)

    return {"status": "updated", "student_id": new_id, "name": new_name}


@app.delete("/profiles/{student_id}")
async def delete_profile(student_id: str):
    """Delete a profile and all its embeddings."""
    gallery = load_gallery()
    key = next((k for k in gallery if str(k) == student_id), None)
    if key is None:
        raise HTTPException(404, f"No profile with ID {student_id}")

    removed = gallery[key]
    gallery_copy = {
        str(sid): {**p, "embeddings": list(p["embeddings"])}
        for sid, p in gallery.items()
        if sid != key
    }
    save_gallery(gallery_copy)
    return {"status": "deleted", "student_id": student_id, "name": removed["name"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Preload model at startup
    get_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
