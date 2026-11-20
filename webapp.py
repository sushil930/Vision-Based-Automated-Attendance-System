"""Web application for the AI group face recognition and attendance system.

A single FastAPI service that serves the browser UI (templates/index.html)
and a JSON API covering the full workflow:

  * Recognize faces in a group photo (MATCH / REVIEW / UNKNOWN + review step)
  * Register new students from faces the system does not know
  * Enrich existing profiles with additional face samples
  * Browse, edit and delete student profiles

The recognition core (model cache, gallery index, batch matcher, thresholds)
is imported from the existing app.py so the matching logic exists exactly
once. This module adds what the original mobile server was missing:

  * profiles.json and gallery/crops are synchronized on EVERY write, so the
    registry never drifts out of sync with the gallery pickle
  * every inference path downscales uploads via preprocess_image()
  * client-supplied embeddings are validated (dimension, norm) and
    re-normalized before they touch the gallery
  * optional token authentication (set ATTENDANCE_API_TOKEN) and opt-in
    CORS (ATTENDANCE_ALLOWED_ORIGINS) instead of allow_origins=["*"]
  * the previous gallery is backed up before every overwrite
  * automatic training: every high-confidence match found during recognition
    is appended to that student's gallery entry (deduplicated, capped per
    student), so recognition keeps improving with every photo analyzed

Run from the project root (next to app.py and gallery/):

    uvicorn webapp:app --host 127.0.0.1 --port 8000

Environment variables (all optional):

    ATTENDANCE_API_TOKEN       if set, all mutating endpoints require the
                               header  X-API-Token: <value>
    ATTENDANCE_ALLOWED_ORIGINS comma-separated origins allowed by CORS
                               (default: none; the UI is same-origin)
    ATTENDANCE_MODEL           insightface model name (default buffalo_l)
    ATTENDANCE_MAX_IMAGE_SIZE  longest-edge downscale limit (default 1280)
    ATTENDANCE_AUTO_LEARN      set to 0 to disable automatic training
    ATTENDANCE_AUTO_LEARN_THRESHOLD    minimum match score to learn from
                                       (default 0.70)
    ATTENDANCE_MAX_SAMPLES_PER_STUDENT gallery cap per student (default 50)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pickle
import re
import shutil
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageOps

LOGGER = logging.getLogger("webapp")

PROJECT_ROOT = Path(__file__).resolve().parent
GALLERY_FILE = PROJECT_ROOT / "gallery" / "multi_sample_gallery.pkl"
PROFILES_FILE = PROJECT_ROOT / "gallery" / "profiles.json"
CROPS_DIR = PROJECT_ROOT / "gallery" / "crops"
BACKUP_DIR = PROJECT_ROOT / "gallery" / "backups"
BACKUPS_TO_KEEP = 5

SUPPORTED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
FACE_KEY_RE = re.compile(r"^face_(\d+)$")

# Enrichment policy (auto-add very confident matches, hold medium ones for
# review). Defined once here; the MATCH/REVIEW/UNKNOWN thresholds come from
# the shared core.
ADD_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.55

# Automatic training: every confident match encountered during recognition is
# appended to that student's gallery entry, so recognition keeps improving
# with use. REVIEW/UNKNOWN faces are never auto-learned - a wrong embedding
# would poison the gallery - they can be taught explicitly via /api/learn
# after a human confirms them in the review step.
AUTO_LEARN = os.environ.get("ATTENDANCE_AUTO_LEARN", "1").lower() not in ("0", "false", "no")
AUTO_LEARN_THRESHOLD = float(os.environ.get("ATTENDANCE_AUTO_LEARN_THRESHOLD", str(ADD_THRESHOLD)))
MAX_SAMPLES_PER_STUDENT = int(os.environ.get("ATTENDANCE_MAX_SAMPLES_PER_STUDENT", "50"))
NEAR_DUPLICATE_SIM = 0.999  # similarity above which a face counts as already stored

API_TOKEN = os.environ.get("ATTENDANCE_API_TOKEN")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ATTENDANCE_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# --------------------------------------------------------------------------
# Shared core (app.py) - imported lazily so gallery/profile management works
# even on a machine where insightface is not installed yet.
# --------------------------------------------------------------------------
_core_module: Any = None
_core_error: str | None = None
_core_lock = threading.Lock()


def get_core():
    """Import and cache app.py. Returns the module or raises HTTPException."""
    global _core_module, _core_error
    with _core_lock:
        if _core_module is not None:
            return _core_module
        if _core_error is not None:
            raise HTTPException(status_code=503, detail=_core_error)
        try:
            import app as core  # noqa: PLC0415 - deliberate lazy import
        except Exception as error:  # pragma: no cover - environment issue
            _core_error = (
                f"Recognition core unavailable: {error}. "
                "Install the project requirements (insightface, onnxruntime, "
                "opencv-python, pillow-heif) next to webapp.py."
            )
            LOGGER.exception("Could not import the recognition core")
            raise HTTPException(status_code=503, detail=_core_error) from error
        _core_module = core
        return core


def warm_up_core() -> None:
    """Load model and gallery index off the request path (startup thread)."""
    try:
        core = get_core()
        core.get_model()
        core.load_gallery_index()
        LOGGER.info("Recognition core warmed up (model + gallery index)")
    except Exception as error:
        LOGGER.warning("Core warm-up skipped: %s", error)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=warm_up_core, daemon=True).start()
    yield


# --------------------------------------------------------------------------
# Authentication (opt-in) and CORS (opt-in, no wildcard default)
# --------------------------------------------------------------------------
async def require_token(request: Request) -> None:
    if API_TOKEN and request.headers.get("X-API-Token") != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Token header")


# Applied to every mutating endpoint below (opt-in via ATTENDANCE_API_TOKEN).
AUTH = [Depends(require_token)]


app = FastAPI(title="AI Attendance Web App", lifespan=lifespan)

if ALLOWED_ORIGINS:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# --------------------------------------------------------------------------
# Gallery + registry persistence
# --------------------------------------------------------------------------
def load_gallery() -> dict[str, dict[str, Any]]:
    if not GALLERY_FILE.is_file():
        raise HTTPException(status_code=500, detail=f"Gallery not found: {GALLERY_FILE}")
    with GALLERY_FILE.open("rb") as gallery_file:
        gallery = pickle.load(gallery_file)
    if not isinstance(gallery, dict) or not gallery:
        raise HTTPException(status_code=500, detail="Gallery is empty or invalid")
    return gallery


def save_gallery(gallery: dict[str, dict[str, Any]]) -> None:
    """Atomically replace the gallery, keeping a dated backup of the old one."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if GALLERY_FILE.is_file():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(GALLERY_FILE, BACKUP_DIR / f"multi_sample_gallery_{stamp}.pkl")
        backups = sorted(BACKUP_DIR.glob("multi_sample_gallery_*.pkl"))
        for old in backups[:-BACKUPS_TO_KEEP]:
            old.unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=GALLERY_FILE.parent, delete=False, suffix=".pkl"
    ) as temporary_file:
        pickle.dump(gallery, temporary_file)
        Path(temporary_file.name).replace(GALLERY_FILE)


def load_profiles() -> dict[str, dict[str, str]]:
    if not PROFILES_FILE.is_file():
        return {}
    with PROFILES_FILE.open("r", encoding="utf-8") as profiles_file:
        return json.load(profiles_file)


def save_profiles(profiles: dict[str, dict[str, str]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=PROFILES_FILE.parent, delete=False, suffix=".json"
    ) as temporary_file:
        json.dump(profiles, temporary_file, indent=4)
        Path(temporary_file.name).replace(PROFILES_FILE)


def embedding_dimension() -> int:
    """Dimension expected for new embeddings, derived from the live gallery."""
    for profile in load_gallery().values():
        embeddings = np.asarray(profile["embeddings"], dtype=np.float32)
        if embeddings.ndim == 2 and len(embeddings):
            return int(embeddings.shape[1])
    return 512


def validate_embedding(raw: Any) -> np.ndarray:
    """Validate a client-supplied embedding and return it L2-normalized."""
    try:
        embedding = np.asarray(raw, dtype=np.float32)
    except Exception as error:
        raise HTTPException(status_code=422, detail="Embedding is not numeric") from error
    if embedding.ndim != 1 or embedding.shape[0] != embedding_dimension():
        raise HTTPException(
            status_code=422,
            detail=f"Embedding must be a 1-D vector of length {embedding_dimension()}",
        )
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm == 0.0:
        raise HTTPException(status_code=422, detail="Embedding has zero or invalid norm")
    return embedding / norm


def learn_embedding(gallery: dict, student_id: str, embedding: np.ndarray) -> dict[str, Any]:
    """Append one normalized embedding to a student's gallery entry.

    Near-duplicates of already stored samples are skipped so re-analyzing
    the same photo does not bloat the gallery. Once a student reaches the
    sample cap, the most redundant stored sample (highest total similarity
    to all others) is replaced by the new one, keeping the gallery diverse.
    """
    profile = gallery[student_id]
    stored = np.asarray(profile["embeddings"], dtype=np.float32)
    if stored.ndim == 1:
        stored = stored[None, :]
    norms = np.linalg.norm(stored, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    stored_unit = stored / norms

    if stored_unit.size and float((stored_unit @ embedding).max()) >= NEAR_DUPLICATE_SIM:
        return {"learned": False, "reason": "duplicate of an existing sample"}

    replaced = False
    if len(profile["embeddings"]) >= MAX_SAMPLES_PER_STUDENT:
        similarity = stored_unit @ stored_unit.T
        np.fill_diagonal(similarity, -1.0)
        profile["embeddings"].pop(int(np.argmax(similarity.sum(axis=1))))
        replaced = True
    profile["embeddings"].append(embedding)
    return {
        "learned": True,
        "reason": "replaced the most redundant sample" if replaced else "new sample stored",
    }


def next_face_key(profiles: dict[str, dict[str, str]]) -> str:
    used = [
        int(match.group(1))
        for key in profiles
        if (match := FACE_KEY_RE.fullmatch(key))
    ]
    return f"face_{max(used, default=0) + 1:02d}"


def registry_add(student_id: str, name: str, crop_b64: str, profiles: dict) -> str:
    """Register metadata + representative crop for one new profile."""
    key = next_face_key(profiles)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    if crop_b64:
        try:
            crop_bytes = base64.b64decode(crop_b64)
            with Image.open(io.BytesIO(crop_bytes)) as crop_image:
                rgb = np.asarray(crop_image.convert("RGB"))
            cv2.imwrite(str(CROPS_DIR / f"{key}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception as error:
            LOGGER.warning("Could not store crop for %s: %s", student_id, error)
    profiles[key] = {"student_id": str(student_id), "name": name}
    return key


def registry_rename(student_id: str, new_student_id: str, new_name: str,
                    gallery: dict, profiles: dict) -> None:
    """Propagate an ID/name edit to the gallery and profiles.json."""
    profile = gallery.pop(str(student_id))
    gallery[str(new_student_id)] = {**profile, "name": new_name}
    for entry in profiles.values():
        if str(entry.get("student_id")) == str(student_id):
            entry["student_id"] = str(new_student_id)
            entry["name"] = new_name


def registry_delete(student_id: str, profiles: dict) -> None:
    """Remove metadata + crop file for one profile."""
    stale_keys = [
        key for key, entry in profiles.items()
        if str(entry.get("student_id")) == str(student_id)
    ]
    for key in stale_keys:
        profiles.pop(key, None)
        crop_path = CROPS_DIR / f"{key}.jpg"
        if crop_path.is_file():
            crop_path.unlink()


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to an orientation-correct RGB array."""
    try:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            pass  # HEIC support optional
        image = Image.open(io.BytesIO(file_bytes))
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {error}") from error


def crop_face(image_rgb: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Create a padded RGB crop around one detected face."""
    x1, y1, x2, y2 = bbox.astype(int)
    px = int((x2 - x1) * 0.3)
    py = int((x2 - y1) * 0.4)
    return image_rgb[
        max(0, y1 - py): min(image_rgb.shape[0], y2 + py),
        max(0, x1 - px): min(image_rgb.shape[1], x2 + px),
    ]


def ndarray_to_b64jpg(arr: np.ndarray, quality: int = 80) -> str:
    """Encode an RGB numpy array as a base64 JPEG string."""
    if arr.size == 0:
        return ""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def detect_and_match(image_rgb: np.ndarray) -> tuple[list, list, np.ndarray, str, list]:
    """Run the shared pipeline: downscale -> detect -> embed -> match.

    Returns (faces, results, resized_rgb, timings_line, normalized_embeddings).
    The normalized embeddings are parallel to faces/results so the automatic
    training, registration and enrichment paths can reuse them instead of
    recomputing. Fixes the original inconsistency where some upload paths
    skipped the downscale step.
    """
    core = get_core()
    timings: list[tuple[str, float]] = []

    t0 = time.perf_counter()
    resized, _scale = core.preprocess_image(image_rgb)
    timings.append(("resize", (time.perf_counter() - t0) * 1000))

    t0 = time.perf_counter()
    faces = core.get_model().get(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    timings.append(("detection", (time.perf_counter() - t0) * 1000))

    t0 = time.perf_counter()
    kept: list = []
    embeddings: list[np.ndarray] = []
    for face in faces:
        embedding = face.embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            continue
        kept.append(face)
        embeddings.append(embedding / norm)
    index = core.load_gallery_index()
    results = core.recognize_faces_batch(embeddings, index)
    timings.append(("matching", (time.perf_counter() - t0) * 1000))

    return kept, results, resized, core.format_timings(timings), embeddings


STATUS_COLORS = {"MATCH": (60, 180, 75), "REVIEW": (230, 160, 30), "UNKNOWN": (200, 60, 60)}


def annotate(resized_rgb: np.ndarray, faces: list, results: list) -> np.ndarray:
    """Draw labeled bounding boxes on the resized image (RGB in, RGB out)."""
    canvas = resized_rgb.copy()
    for face, result in zip(faces, results):
        x1, y1, x2, y2 = face.bbox.astype(int)
        color = STATUS_COLORS.get(result["status"], (128, 128, 128))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{result['name']} {result['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, max(11, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "api_token_required": bool(API_TOKEN),
    })


# --------------------------------------------------------------------------
# Read-only API
# --------------------------------------------------------------------------
@app.get("/api/stats")
def stats():
    try:
        gallery = load_gallery()
    except HTTPException:
        return {"gallery_ok": False, "students": 0, "total_samples": 0}
    samples = sum(len(p["embeddings"]) for p in gallery.values())
    core_ready = _core_module is not None or _core_error is None
    thresholds = {"match": None, "min_margin": None}
    if core_ready:
        try:
            core = get_core()
            thresholds = {"match": core.MATCH_THRESHOLD, "min_margin": core.MIN_MARGIN}
            core_ready = True
        except HTTPException:
            core_ready = False
    return {
        "gallery_ok": True,
        "students": len(gallery),
        "total_samples": samples,
        "model": os.environ.get("ATTENDANCE_MODEL", "buffalo_l"),
        "core_ready": core_ready,
        "thresholds": thresholds,
        "add_threshold": ADD_THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        "auto_learn": AUTO_LEARN,
        "auto_learn_threshold": AUTO_LEARN_THRESHOLD,
        "max_samples_per_student": MAX_SAMPLES_PER_STUDENT,
    }


@app.get("/api/profiles")
def list_profiles():
    gallery = load_gallery()
    profiles = load_profiles()
    id_to_key = {str(v["student_id"]): k for k, v in profiles.items()}
    rows = [
        {
            "student_id": str(sid),
            "name": profile["name"],
            "samples": len(profile["embeddings"]),
            "face_key": id_to_key.get(str(sid)),
        }
        for sid, profile in sorted(gallery.items(), key=lambda item: str(item[0]))
    ]
    return {"profiles": rows}


@app.get("/api/crops/{face_key}")
def crop_image(face_key: str):
    if not FACE_KEY_RE.fullmatch(face_key):
        raise HTTPException(status_code=400, detail="Invalid crop key")
    path = CROPS_DIR / f"{face_key}.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(path, media_type="image/jpeg")


# --------------------------------------------------------------------------
# Recognition
# --------------------------------------------------------------------------
@app.post("/api/recognize", dependencies=AUTH)
async def recognize(image: UploadFile = File(...)):
    image_rgb = decode_upload(await image.read())
    faces, results, resized, timings, embeddings = detect_and_match(image_rgb)

    # Automatic training: append every high-confidence match to that
    # student's gallery entry, then persist once if anything changed. The
    # dedupe/cap logic in learn_embedding keeps the gallery clean.
    learned: list[dict[str, Any]] = []
    if AUTO_LEARN and embeddings:
        core = get_core()
        gallery = load_gallery()
        changed = False
        for result, embedding in zip(results, embeddings):
            if (result["status"] != "MATCH"
                    or result["score"] < AUTO_LEARN_THRESHOLD
                    or result["margin"] < core.MIN_MARGIN
                    or result["student_id"] not in gallery):
                continue
            info = learn_embedding(gallery, result["student_id"], embedding)
            changed |= info["learned"]
            learned.append({
                "student_id": result["student_id"],
                "name": result["name"],
                "score": round(result["score"], 4),
                **info,
            })
        if changed:
            save_gallery(gallery)

    annotated_b64 = ""
    face_rows = []
    for i, (face, result, embedding) in enumerate(zip(faces, results, embeddings)):
        face_rows.append({
            "index": i,
            "student_id": result["student_id"] if result["status"] != "UNKNOWN" else "",
            "name": result["name"] if result["status"] != "UNKNOWN" else "",
            "score": round(result["score"], 4),
            "margin": round(result["margin"], 4),
            "status": result["status"],
            "bbox": face.bbox.astype(int).tolist(),
            "crop_b64": ndarray_to_b64jpg(crop_face(resized, face.bbox), quality=70),
            "embedding": embedding.tolist(),
        })
    if faces:
        annotated_b64 = ndarray_to_b64jpg(annotate(resized, faces, results), quality=85)

    counts = {
        "match": sum(1 for r in results if r["status"] == "MATCH"),
        "review": sum(1 for r in results if r["status"] == "REVIEW"),
        "unknown": sum(1 for r in results if r["status"] == "UNKNOWN"),
    }
    return {"faces": face_rows, "counts": counts, "annotated_b64": annotated_b64,
            "timings": timings, "learned": learned,
            "auto_learn": AUTO_LEARN, "auto_learn_threshold": AUTO_LEARN_THRESHOLD}


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
@app.post("/api/register", dependencies=AUTH)
async def register(image: UploadFile = File(...)):
    """Return every face in the photo that is NOT a confident gallery match."""
    image_rgb = decode_upload(await image.read())
    faces, results, resized, _timings, embeddings = detect_and_match(image_rgb)

    candidates = []
    for i, (face, result, embedding) in enumerate(zip(faces, results, embeddings)):
        if result["status"] == "MATCH":
            continue
        candidates.append({
            "index": i,
            "best_match_name": result["name"] if result["status"] != "UNKNOWN" else "",
            "best_score": round(result["score"], 4),
            "status": result["status"],
            "embedding": embedding.tolist(),
            "crop_b64": ndarray_to_b64jpg(crop_face(resized, face.bbox), quality=85),
        })
    return {"candidates": candidates}


@app.post("/api/register/save", dependencies=AUTH)
async def register_save(request: Request):
    body = await _json_body(request)
    new_entries = body.get("new", [])
    appends = body.get("appends", [])
    if not new_entries and not appends:
        raise HTTPException(status_code=422, detail="Nothing to save")

    gallery = load_gallery()
    profiles = load_profiles()
    existing_ids = {str(p["student_id"]) for p in profiles.values()}
    existing_names = {p["name"].casefold() for p in profiles.values()}

    # Phase 1: validate everything before touching disk.
    prepared_new = []
    for entry in new_entries:
        student_id = str(entry.get("student_id", "")).strip()
        name = str(entry.get("name", "")).strip()
        if not student_id or not name:
            raise HTTPException(status_code=422, detail="Every new profile needs a student ID and a name")
        if student_id in existing_ids or student_id in gallery:
            raise HTTPException(status_code=409, detail=f"Student ID {student_id} already exists")
        prepared_new.append((student_id, name, validate_embedding(entry.get("embedding")),
                             entry.get("crop_b64", "")))
    prepared_appends = []
    for entry in appends:
        student_id = str(entry.get("student_id", "")).strip()
        name = str(entry.get("name", "")).strip()
        if student_id not in gallery:
            raise HTTPException(status_code=404, detail=f"Unknown student ID {student_id}")
        if name and name.casefold() != gallery[student_id]["name"].casefold():
            raise HTTPException(status_code=409, detail=(
                f"Name mismatch for ID {student_id}: gallery has "
                f"\"{gallery[student_id]['name']}\", got \"{name}\""))
        if not name:
            name = gallery[student_id]["name"]
        prepared_appends.append((student_id, validate_embedding(entry.get("embedding"))))

    # Phase 2: apply - gallery entries, crops, profiles.json, one atomic write.
    added, appended = [], []
    for student_id, name, embedding, crop_b64 in prepared_new:
        gallery[student_id] = {"name": name, "embeddings": [embedding]}
        key = registry_add(student_id, name, crop_b64, profiles)
        added.append({"student_id": student_id, "name": name, "face_key": key})
    for student_id, embedding in prepared_appends:
        gallery[student_id]["embeddings"].append(embedding)
        appended.append(student_id)

    save_profiles(profiles)
    save_gallery(gallery)
    return {"added": added, "appended": appended,
            "total_students": len(gallery)}


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from error
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return body


# --------------------------------------------------------------------------
# Sample enrichment
# --------------------------------------------------------------------------
@app.post("/api/add-samples", dependencies=AUTH)
async def add_samples(request: Request):
    """Propose additions from uploaded enrollment photos.

    Matches >= ADD_THRESHOLD are pre-selected (auto_add); scores between
    REVIEW_THRESHOLD and ADD_THRESHOLD are offered for manual selection.
    """
    form = await request.form()
    # multi_items(), not values(): repeated field names must all be kept.
    uploads = [value for _key, value in form.multi_items() if hasattr(value, "read")]
    if not uploads:
        raise HTTPException(status_code=422, detail="No images uploaded")

    proposals: list[dict[str, Any]] = []
    for upload in uploads:
        image_rgb = decode_upload(await upload.read())
        faces, results, resized, _timings, embeddings = detect_and_match(image_rgb)
        for face, result, embedding in zip(faces, results, embeddings):
            score = result["score"]
            if score < REVIEW_THRESHOLD:
                continue
            proposals.append({
                "student_id": result["student_id"],
                "name": result["name"],
                "score": round(score, 4),
                "auto_add": score >= ADD_THRESHOLD,
                "embedding": embedding.tolist(),
                "crop_b64": ndarray_to_b64jpg(crop_face(resized, face.bbox), quality=70),
            })
    return {"proposals": proposals,
            "add_threshold": ADD_THRESHOLD, "review_threshold": REVIEW_THRESHOLD}


@app.post("/api/add-samples/commit", dependencies=AUTH)
async def add_samples_commit(request: Request):
    body = await _json_body(request)
    additions = body.get("additions", [])
    if not additions:
        raise HTTPException(status_code=422, detail="No additions selected")

    gallery = load_gallery()
    prepared = []
    per_student: dict[str, int] = {}
    for entry in additions:
        student_id = str(entry.get("student_id", "")).strip()
        if student_id not in gallery:
            raise HTTPException(status_code=404, detail=f"Unknown student ID {student_id}")
        prepared.append((student_id, validate_embedding(entry.get("embedding"))))
    for student_id, embedding in prepared:
        gallery[student_id]["embeddings"].append(embedding)
        per_student[student_id] = per_student.get(student_id, 0) + 1
    save_gallery(gallery)
    return {"added_samples": {sid: n for sid, n in sorted(per_student.items())},
            "total_samples": sum(len(p["embeddings"]) for p in gallery.values())}


@app.post("/api/learn", dependencies=AUTH)
async def learn_sample(request: Request):
    """Teach the system one user-confirmed face (Recognize review step).

    The client sends the embedding of a REVIEW face a human has confirmed;
    it is deduplicated, appended to that student's gallery entry and
    persisted. UNKNOWN faces carry no student ID and cannot be learned here.
    """
    body = await _json_body(request)
    student_id = str(body.get("student_id", "")).strip()
    if not student_id:
        raise HTTPException(status_code=422, detail="student_id is required")
    gallery = load_gallery()
    if student_id not in gallery:
        raise HTTPException(status_code=404, detail=f"Unknown student ID {student_id}")
    embedding = validate_embedding(body.get("embedding"))
    info = learn_embedding(gallery, student_id, embedding)
    if info["learned"]:
        save_gallery(gallery)
    return {"student_id": student_id, "name": gallery[student_id]["name"],
            "samples": len(gallery[student_id]["embeddings"]), **info}


# --------------------------------------------------------------------------
# Profile administration
# --------------------------------------------------------------------------
@app.put("/api/profiles/{student_id}", dependencies=AUTH)
async def edit_profile(student_id: str, request: Request):
    body = await _json_body(request)
    new_name = str(body.get("name", "")).strip()
    new_id = str(body.get("new_student_id", student_id)).strip()
    if not new_name:
        raise HTTPException(status_code=422, detail="A name is required")

    gallery = load_gallery()
    profiles = load_profiles()
    if student_id not in gallery:
        raise HTTPException(status_code=404, detail=f"Unknown student ID {student_id}")
    if new_id != student_id and (new_id in gallery or
                                  new_id in {str(p["student_id"]) for p in profiles.values()}):
        raise HTTPException(status_code=409, detail=f"Student ID {new_id} already exists")

    registry_rename(student_id, new_id, new_name, gallery, profiles)
    save_profiles(profiles)
    save_gallery(gallery)
    return {"student_id": new_id, "name": new_name,
            "samples": len(gallery[new_id]["embeddings"])}


@app.delete("/api/profiles/{student_id}", dependencies=AUTH)
def delete_profile(student_id: str):
    gallery = load_gallery()
    profiles = load_profiles()
    if student_id not in gallery:
        raise HTTPException(status_code=404, detail=f"Unknown student ID {student_id}")

    removed_name = gallery[student_id]["name"]
    del gallery[student_id]
    registry_delete(student_id, profiles)
    save_profiles(profiles)
    save_gallery(gallery)
    return {"deleted": student_id, "name": removed_name, "remaining": len(gallery)}




if __name__ == "__main__":
    uvicorn.run("webapp:app", host=os.environ.get("ATTENDANCE_HOST", "127.0.0.1"),
                port=int(os.environ.get("ATTENDANCE_PORT", "8000")))
