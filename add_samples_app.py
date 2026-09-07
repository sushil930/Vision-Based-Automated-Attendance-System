"""Gradio interface for adding new face samples to the gallery from images."""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from app import GALLERY_FILE, PROJECT_ROOT, get_model, load_gallery, recognize_face

LOGGER = logging.getLogger(__name__)
ADD_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.55

register_heif_opener()

def decode_image(image_path: str) -> np.ndarray:
    """Load a standard, HEIC, or HEIF image as an orientation-correct RGB array."""
    with Image.open(image_path) as uploaded_image:
        return np.asarray(ImageOps.exif_transpose(uploaded_image).convert("RGB"))

def save_gallery(gallery: dict[str, dict[str, Any]]) -> None:
    """Atomically replace the multi-sample gallery after all entries are valid."""
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=GALLERY_FILE.parent, delete=False, suffix=".pkl"
    ) as temporary_file:
        pickle.dump(gallery, temporary_file)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(GALLERY_FILE)

def process_images(files: list[str]) -> tuple[str, list, list, list, gr.update, gr.update]:
    if not files:
        return "No images uploaded.", [], [], [], gr.update(choices=[], value=[]), gr.update(interactive=False)

    try:
        gallery = load_gallery()
    except Exception as error:
        return f"Error loading gallery: {error}", [], [], [], gr.update(choices=[], value=[]), gr.update(interactive=False)

    model = get_model()
    
    proposed_additions = []
    review_choices = []
    review_cards = []
    default_selected = []
    
    total_faces = 0
    auto_add_count = 0
    review_count = 0
    unknown_count = 0

    face_index = 0

    for file_path in files:
        try:
            image_rgb = decode_image(file_path)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except Exception as error:
            LOGGER.warning("Could not read image %s: %s", file_path, error)
            continue
            
        faces = model.get(image_bgr)
        total_faces += len(faces)
        
        for face in faces:
            embedding = face.embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                continue
            embedding /= norm
            
            result = recognize_face(embedding, gallery)
            score = result["score"]
            match_name = result["name"]
            student_id = result["student_id"]
            
            if score >= REVIEW_THRESHOLD:
                face_index += 1
                
                # Create a crop
                x1, y1, x2, y2 = face.bbox.astype(int)
                padding_x = int((x2 - x1) * 0.3)
                padding_y = int((y2 - y1) * 0.4)
                crop = image_rgb[
                    max(0, y1 - padding_y) : min(image_rgb.shape[0], y2 + padding_y),
                    max(0, x1 - padding_x) : min(image_rgb.shape[1], x2 + padding_x),
                ]
                
                choice_label = f"Face {face_index}: {match_name} (ID: {student_id}, Score: {score:.3f})"
                
                proposed_additions.append({
                    "id": choice_label,
                    "student_id": student_id,
                    "embedding": embedding
                })
                
                review_choices.append(choice_label)
                review_cards.append((crop, choice_label))
                
                if score >= ADD_THRESHOLD:
                    default_selected.append(choice_label)
                    auto_add_count += 1
                else:
                    review_count += 1
            else:
                unknown_count += 1

    summary = (
        f"Processed {len(files)} images. **{total_faces} faces detected**.\n\n"
        f"**{auto_add_count}** confident matches (auto-selected).\n"
        f"**{review_count}** uncertain matches (need review).\n"
        f"**{unknown_count}** unknown faces ignored."
    )
    
    return (
        summary,
        proposed_additions,
        review_cards,
        review_choices,
        gr.update(choices=review_choices, value=default_selected),
        gr.update(interactive=len(review_choices) > 0)
    )

def commit_additions(
    proposed: list[dict[str, Any]] | None,
    confirmed: list[str]
) -> tuple[str, list, list, list, gr.update, gr.update]:
    if not proposed or not confirmed:
        return "No samples selected to add.", [], [], [], gr.update(choices=[], value=[]), gr.update(interactive=False)
        
    try:
        gallery = load_gallery()
    except Exception as error:
        return f"Error loading gallery: {error}", [], [], [], gr.update(choices=[], value=[]), gr.update(interactive=False)

    added_count = 0
    confirmed_set = set(confirmed)
    
    for item in proposed:
        if item["id"] in confirmed_set:
            student_id = item["student_id"]
            if student_id in gallery:
                gallery[student_id]["embeddings"].append(item["embedding"])
                added_count += 1
                
    if added_count > 0:
        save_gallery(gallery)
        msg = f"Successfully added {added_count} new samples to the gallery."
    else:
        msg = "No samples were added."
        
    # Clear UI state
    return msg, [], [], [], gr.update(choices=[], value=[]), gr.update(interactive=False)

def create_app() -> gr.Blocks:
    with gr.Blocks(title="AI Add Face Samples") as interface:
        gr.Markdown(
            "# Add Face Samples\n"
            "Upload images to extract face samples and add them to existing student profiles to improve recognition accuracy."
        )
        
        proposed_state = gr.State([])
        
        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.File(
                    label="Enrollment Images (JPG, PNG, HEIC, or HEIF)",
                    file_count="multiple",
                    file_types=[".jpg", ".jpeg", ".png", ".heic", ".heif"],
                    type="filepath",
                )
                process_button = gr.Button("Process Images", variant="primary")
                summary = gr.Markdown("Upload images and click Process to begin.")
                
            with gr.Column(scale=1):
                gr.Markdown("### Review Matches")
                gr.Markdown(
                    "Matches with score >= 0.70 are auto-selected. Matches between 0.55 and 0.70 require manual selection. "
                    "Verify the faces and click the button below to add them to the gallery."
                )
                review_gallery = gr.Gallery(
                    label="Detected Faces",
                    columns=4,
                    height="auto",
                    object_fit="contain",
                )
                review_checkboxes = gr.CheckboxGroup(
                    label="Confirm Faces to Add",
                    choices=[],
                )
                add_button = gr.Button("Add Selected Samples to Gallery", variant="primary", interactive=False)
                
        process_button.click(
            fn=process_images,
            inputs=[uploader],
            outputs=[
                summary,
                proposed_state,
                review_gallery,
                review_checkboxes,
                review_checkboxes,
                add_button
            ]
        )
        
        add_button.click(
            fn=commit_additions,
            inputs=[proposed_state, review_checkboxes],
            outputs=[
                summary,
                proposed_state,
                review_gallery,
                review_checkboxes,
                review_checkboxes,
                add_button
            ]
        )
        
    return interface

if __name__ == "__main__":
    app = create_app()
    app.launch()
