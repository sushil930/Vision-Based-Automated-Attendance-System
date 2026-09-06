"""Gradio web interface for group face recognition and attendance review."""

from __future__ import annotations

import logging
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from insightface.app import FaceAnalysis


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
GALLERY_FILE = PROJECT_ROOT / "gallery" / "multi_sample_gallery.pkl"
MATCH_THRESHOLD = 0.55
MIN_MARGIN = 0.05
DETECTION_SIZE = (640, 640)


# Register HEIC/HEIF support before Gradio or Pillow opens uploaded images.
register_heif_opener()


@lru_cache(maxsize=1)
def get_model() -> FaceAnalysis:
    """Load the InsightFace model once per running web application."""
    LOGGER.info("Loading InsightFace model")
    model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=0, det_size=DETECTION_SIZE)
    return model


def load_gallery() -> dict[str, dict[str, Any]]:
    """Load the current multi-sample gallery from the project directory."""
    if not GALLERY_FILE.is_file():
        raise FileNotFoundError(f"Gallery not found: {GALLERY_FILE}")

    with GALLERY_FILE.open("rb") as gallery_file:
        gallery = pickle.load(gallery_file)

    if not isinstance(gallery, dict) or not gallery:
        raise ValueError("The gallery is empty or is not a multi-sample gallery.")

    return gallery


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity, safely handling zero-length embeddings."""
    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)
    if first_norm == 0 or second_norm == 0:
        return -1.0
    return float(np.dot(first, second) / (first_norm * second_norm))


def recognize_face(
    face_embedding: np.ndarray, gallery: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Classify one detected face against every stored gallery embedding."""
    embedding = face_embedding.astype(np.float32)
    person_results: list[dict[str, Any]] = []

    for student_id, profile in gallery.items():
        best_score = max(
            (
                cosine_similarity(
                    embedding, np.asarray(stored_embedding, dtype=np.float32)
                )
                for stored_embedding in profile["embeddings"]
            ),
            default=-1.0,
        )
        person_results.append(
            {
                "student_id": str(student_id),
                "name": profile["name"],
                "score": best_score,
            }
        )

    person_results.sort(key=lambda result: result["score"], reverse=True)
    best = person_results[0]
    margin = best["score"] - person_results[1]["score"] if len(person_results) > 1 else 1.0

    if best["score"] >= MATCH_THRESHOLD and margin >= MIN_MARGIN:
        status = "MATCH"
    elif best["score"] >= 0.45:
        status = "REVIEW"
    else:
        status = "UNKNOWN"

    return {**best, "margin": margin, "status": status}


def recognize_group_image(
    uploaded_file: str | None,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, str]], list[list[str]], str]:
    """Recognize uploaded group-photo faces and prepare Gradio display data."""
    if uploaded_file is None:
        return None, [], [], "Please upload a JPG, PNG, HEIC, or HEIF group photo."

    try:
        with Image.open(uploaded_file) as uploaded_image:
            original_rgb = np.asarray(
                ImageOps.exif_transpose(uploaded_image).convert("RGB")
            )
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr = original_bgr.copy()
        faces = get_model().get(annotated_bgr)
        gallery = load_gallery()
    except Exception as error:  # Keep a bad file or model issue from taking down the UI.
        LOGGER.exception("Could not process uploaded image")
        return None, [], [], f"Could not process the image: {error}"

    if not faces:
        return (
            original_rgb,
            [],
            [],
            "No faces detected. Try a sharper, well-lit group photo.",
        )

    cards: list[tuple[np.ndarray, str]] = []
    table_rows: list[list[str]] = []
    counts = {"MATCH": 0, "REVIEW": 0, "UNKNOWN": 0}

    for index, face in enumerate(faces, start=1):
        result = recognize_face(face.embedding, gallery)
        counts[result["status"]] += 1
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2 = min(x2, annotated_bgr.shape[1])
        y2 = min(y2, annotated_bgr.shape[0])

        colors = {
            "MATCH": (0, 180, 0),
            "REVIEW": (0, 180, 255),
            "UNKNOWN": (0, 0, 220),
        }
        color = colors[result["status"]]
        display_name = result["name"] if result["status"] != "UNKNOWN" else "Unknown"
        label = f"{display_name} {result['score']:.2f}"
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated_bgr,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        padding_x = int((x2 - x1) * 0.3)
        padding_y = int((y2 - y1) * 0.4)
        crop = original_rgb[
            max(0, y1 - padding_y) : min(original_rgb.shape[0], y2 + padding_y),
            max(0, x1 - padding_x) : min(original_rgb.shape[1], x2 + padding_x),
        ]
        cards.append((crop, f"{display_name} - {result['status']} ({result['score']:.3f})"))
        table_rows.append(
            [
                str(index),
                display_name,
                result["student_id"] if result["status"] != "UNKNOWN" else "-",
                result["status"],
                f"{result['score']:.3f}",
            ]
        )

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    summary = (
        f"**{len(faces)} faces detected**  |  "
        f"**{counts['MATCH']} recognized**  |  "
        f"**{counts['REVIEW']} need review**  |  "
        f"**{counts['UNKNOWN']} unknown**"
    )
    return annotated_rgb, cards, table_rows, summary


def clear_results() -> tuple[None, None, list[tuple[np.ndarray, str]], list[list[str]], str]:
    """Reset every interface component to its initial state."""
    return None, None, [], [], "Upload a group photo to begin."


def create_app() -> gr.Blocks:
    """Create the Gradio interface without starting a server."""
    with gr.Blocks(title="AI Group Face Recognition") as interface:
        gr.Markdown(
            "# AI Group Face Recognition\n"
            "Upload a group photo to identify registered students and review attendance."
        )
        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.File(
                    label="Group photo (JPG, PNG, HEIC, or HEIF)",
                    file_types=[".jpg", ".jpeg", ".png", ".heic", ".heif"],
                    type="filepath",
                )
                with gr.Row():
                    recognize_button = gr.Button("Recognize Faces", variant="primary")
                    clear_button = gr.Button("Clear")
                summary = gr.Markdown("Upload a group photo to begin.")
            with gr.Column(scale=1):
                annotated_image = gr.Image(label="Annotated group photo", type="numpy")

        gr.Markdown("## Detected faces")
        face_gallery = gr.Gallery(label="Recognition results", columns=4, object_fit="cover")
        results_table = gr.Dataframe(
            headers=["Face", "Name", "Student ID", "Status", "Confidence"],
            datatype=["str", "str", "str", "str", "str"],
            label="Recognition details",
            interactive=False,
        )

        recognize_button.click(
            recognize_group_image,
            inputs=uploader,
            outputs=[annotated_image, face_gallery, results_table, summary],
        )
        clear_button.click(
            clear_results,
            outputs=[uploader, annotated_image, face_gallery, results_table, summary],
        )

    return interface


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    create_app().launch()
