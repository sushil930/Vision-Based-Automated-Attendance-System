"""Gradio interface for registering every face detected in a group image."""

from __future__ import annotations

import json
import logging
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from app import GALLERY_FILE, PROJECT_ROOT, get_model, load_gallery, recognize_face


LOGGER = logging.getLogger(__name__)
PROFILES_FILE = PROJECT_ROOT / "gallery" / "profiles.json"
CROPS_DIR = PROJECT_ROOT / "gallery" / "crops"
LEGACY_GALLERY_FILE = PROJECT_ROOT / "gallery" / "face_gallery.pkl"
SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".heic", ".heif"]


def profile_rows() -> list[list[str]]:
    """Format all live gallery profiles for the read-only profile table."""
    try:
        gallery = load_gallery()
    except Exception as error:
        LOGGER.warning("Could not load the profile list: %s", error)
        return []
    return [
        [str(student_id), profile["name"], str(len(profile["embeddings"]))]
        for student_id, profile in sorted(gallery.items(), key=lambda item: str(item[0]))
    ]


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize an InsightFace embedding before storing it."""
    normalized = embedding.astype(np.float32)
    length = np.linalg.norm(normalized)
    if length == 0:
        raise ValueError("The detected face has an invalid embedding.")
    return normalized / length


def decode_image(image_path: str) -> np.ndarray:
    """Load a standard, HEIC, or HEIF image as an orientation-correct RGB array."""
    with Image.open(image_path) as uploaded_image:
        return np.asarray(ImageOps.exif_transpose(uploaded_image).convert("RGB"))


def face_crop(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Create a padded RGB crop around one detected face."""
    x1, y1, x2, y2 = bbox.astype(int)
    padding_x = int((x2 - x1) * 0.3)
    padding_y = int((y2 - y1) * 0.4)
    return image[
        max(0, y1 - padding_y) : min(image.shape[0], y2 + padding_y),
        max(0, x1 - padding_x) : min(image.shape[1], x2 + padding_x),
    ]


def save_gallery(gallery: dict[str, dict[str, Any]]) -> None:
    """Atomically replace the multi-sample gallery after all entries are valid."""
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=GALLERY_FILE.parent, delete=False, suffix=".pkl"
    ) as temporary_file:
        pickle.dump(gallery, temporary_file)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(GALLERY_FILE)


def save_new_profile_registries(new_profiles: list[dict[str, Any]]) -> None:
    """Store metadata and one representative crop for every newly created profile."""
    if not new_profiles:
        return

    profiles: dict[str, dict[str, str]] = {}
    if PROFILES_FILE.exists():
        with PROFILES_FILE.open("r", encoding="utf-8") as profiles_file:
            profiles = json.load(profiles_file)

    existing_ids = {profile["student_id"] for profile in profiles.values()}
    used_numbers = [
        int(match.group(1))
        for profile_key in profiles
        if (match := re.fullmatch(r"face_(\d+)", profile_key))
    ]
    next_number = max(used_numbers, default=0) + 1
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    for profile in new_profiles:
        if profile["student_id"] in existing_ids:
            continue
        profile_key = f"face_{next_number:02d}"
        next_number += 1
        Image.fromarray(profile["crop"]).save(
            CROPS_DIR / f"{profile_key}.jpg", format="JPEG"
        )
        profiles[profile_key] = {
            "student_id": profile["student_id"],
            "name": profile["name"],
        }

    with PROFILES_FILE.open("w", encoding="utf-8") as profiles_file:
        json.dump(profiles, profiles_file, indent=4)
        profiles_file.write("\n")


def detect_group_faces(
    uploaded_file: str | None,
) -> tuple[np.ndarray | None, list[tuple[np.ndarray, str]], list[list[str]], list[dict[str, Any]], str]:
    """Detect group-photo faces and prepare registration rows only for new faces."""
    if not uploaded_file:
        return None, [], [], [], "Upload a group photo first."

    try:
        original_rgb = decode_image(uploaded_file)
        annotated_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        faces = get_model().get(annotated_bgr)
        gallery = load_gallery()
    except Exception as error:
        LOGGER.exception("Could not detect faces")
        return None, [], [], [], f"Could not process the image: {error}"

    if not faces:
        return original_rgb, [], [], [], "No faces detected. Use a clearer, well-lit group photo."

    faces = sorted(faces, key=lambda face: (face.bbox[1], face.bbox[0]))
    detections: list[dict[str, Any]] = []
    preview_cards: list[tuple[np.ndarray, str]] = []
    assignment_rows: list[list[str]] = []
    registered_count = 0

    for face in faces:
        match = recognize_face(face.embedding, gallery)
        if match["status"] == "MATCH":
            registered_count += 1
            continue

        label = f"Face {len(detections) + 1}"
        x1, y1, x2, y2 = face.bbox.astype(int)
        color = (255, 180, 0)
        cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated_bgr,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        crop = face_crop(original_rgb, face.bbox)
        detections.append(
            {
                "label": label,
                "embedding": normalize_embedding(face.embedding),
                "crop": crop,
            }
        )
        preview_cards.append((crop, label))
        assignment_rows.append([label, "", ""])

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    if not detections:
        return (
            original_rgb,
            [],
            [],
            [],
            f"All **{registered_count} detected faces** already have registered profiles. "
            "There are no new faces to add.",
        )

    return (
        annotated_rgb,
        preview_cards,
        assignment_rows,
        detections,
        f"Detected **{len(faces)} faces**: **{registered_count} registered** face(s) hidden and "
        f"**{len(detections)} unregistered** face(s) ready to add. Enter a student ID and name for each new face.",
    )


def coerce_rows(assignment_table: Any) -> list[list[str]]:
    """Accept Gradio's Dataframe value in array, pandas, or list form."""
    if assignment_table is None:
        return []
    if hasattr(assignment_table, "values"):
        assignment_table = assignment_table.values.tolist()
    return [[str(cell).strip() for cell in row] for row in assignment_table]


def save_assignments(
    detections: list[dict[str, Any]] | None, assignment_table: Any
) -> tuple[list[list[str]], str]:
    """Save named detections as new profiles or append samples to known profiles."""
    if not detections:
        return profile_rows(), "Detect faces in a group photo before saving assignments."

    rows = coerce_rows(assignment_table)
    if len(rows) != len(detections):
        return profile_rows(), "The assignment table no longer matches the detected faces. Detect again."

    assignments: list[dict[str, Any]] = []
    ids_in_submission: set[str] = set()
    for detection, row in zip(detections, rows):
        student_id = row[1] if len(row) > 1 else ""
        name = row[2] if len(row) > 2 else ""
        if not student_id and not name:
            continue
        if not student_id or not name:
            return profile_rows(), f"{detection['label']} needs both a student ID and a name."
        if student_id in ids_in_submission:
            return profile_rows(), f"Student ID {student_id} is assigned to more than one face."
        ids_in_submission.add(student_id)
        assignments.append({**detection, "student_id": student_id, "name": name})

    if not assignments:
        return profile_rows(), "Enter at least one complete student ID and name before saving."

    gallery = load_gallery()
    gallery_to_save = {
        str(student_id): {**profile, "embeddings": list(profile["embeddings"])}
        for student_id, profile in gallery.items()
    }
    new_profiles: list[dict[str, Any]] = []

    for assignment in assignments:
        existing_profile = gallery_to_save.get(assignment["student_id"])
        if existing_profile:
            if existing_profile["name"].casefold() != assignment["name"].casefold():
                return (
                    profile_rows(),
                    f"Student ID {assignment['student_id']} already belongs to "
                    f"{existing_profile['name']}. Use the existing name to add another sample.",
                )
            existing_profile["embeddings"].append(assignment["embedding"])
        else:
            gallery_to_save[assignment["student_id"]] = {
                "student_id": assignment["student_id"],
                "name": assignment["name"],
                "embeddings": [assignment["embedding"]],
            }
            new_profiles.append(assignment)

    try:
        save_gallery(gallery_to_save)
        save_new_profile_registries(new_profiles)
    except Exception as error:
        LOGGER.exception("Could not save profile assignments")
        return profile_rows(), f"The assignments could not be saved: {error}"

    return (
        profile_rows(),
        f"Saved **{len(assignments)} profile assignment(s)** from this group photo. "
        f"Created {len(new_profiles)} new profile(s).",
    )


def update_profile_registry(old_student_id: str, new_student_id: str, new_name: str) -> None:
    """Keep the metadata registry aligned with a renamed live-gallery profile."""
    if not PROFILES_FILE.exists():
        return

    with PROFILES_FILE.open("r", encoding="utf-8") as profiles_file:
        profiles = json.load(profiles_file)
    for profile in profiles.values():
        if str(profile["student_id"]) == old_student_id:
            profile["student_id"] = new_student_id
            profile["name"] = new_name
    with PROFILES_FILE.open("w", encoding="utf-8") as profiles_file:
        json.dump(profiles, profiles_file, indent=4)
        profiles_file.write("\n")


def update_legacy_gallery(old_student_id: str, new_student_id: str, new_name: str) -> None:
    """Synchronize the older single-sample gallery when it is present."""
    if not LEGACY_GALLERY_FILE.exists():
        return

    with LEGACY_GALLERY_FILE.open("rb") as gallery_file:
        legacy_gallery = pickle.load(gallery_file)
    if not isinstance(legacy_gallery, list):
        return

    for profile in legacy_gallery:
        if str(profile.get("student_id")) == old_student_id:
            profile["student_id"] = new_student_id
            profile["name"] = new_name

    with tempfile.NamedTemporaryFile(
        mode="wb", dir=LEGACY_GALLERY_FILE.parent, delete=False, suffix=".pkl"
    ) as temporary_file:
        pickle.dump(legacy_gallery, temporary_file)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(LEGACY_GALLERY_FILE)


def edit_profile(
    current_student_id: str, updated_student_id: str, updated_name: str
) -> tuple[list[list[str]], str]:
    """Rename a profile or change its student ID while preserving all embeddings."""
    current_student_id = current_student_id.strip()
    updated_student_id = updated_student_id.strip()
    updated_name = updated_name.strip()
    if not current_student_id or not updated_student_id or not updated_name:
        return profile_rows(), "Enter the current ID, new ID, and updated name."

    gallery = load_gallery()
    current_key = next(
        (key for key in gallery if str(key) == current_student_id), None
    )
    if current_key is None:
        return profile_rows(), f"No profile exists with student ID {current_student_id}."

    new_key = next(
        (key for key in gallery if str(key) == updated_student_id), None
    )
    if new_key is not None and new_key != current_key:
        return profile_rows(), f"Student ID {updated_student_id} is already in use."

    gallery_to_save = {
        str(student_id): {**profile, "embeddings": list(profile["embeddings"])}
        for student_id, profile in gallery.items()
    }
    profile = gallery_to_save.pop(str(current_key))
    profile["student_id"] = updated_student_id
    profile["name"] = updated_name
    gallery_to_save[updated_student_id] = profile

    try:
        save_gallery(gallery_to_save)
        update_profile_registry(current_student_id, updated_student_id, updated_name)
        update_legacy_gallery(current_student_id, updated_student_id, updated_name)
    except Exception as error:
        LOGGER.exception("Could not update the profile")
        return profile_rows(), f"The profile could not be updated: {error}"

    return (
        profile_rows(),
        f"Updated profile {current_student_id} to **{updated_name} ({updated_student_id})**. "
        "All saved face samples were preserved.",
    )


def remove_profile_registry(student_id: str) -> None:
    """Remove matching registry entries and their representative face crops."""
    if not PROFILES_FILE.exists():
        return

    with PROFILES_FILE.open("r", encoding="utf-8") as profiles_file:
        profiles = json.load(profiles_file)
    profile_keys = [
        key
        for key, profile in profiles.items()
        if str(profile.get("student_id")) == student_id
    ]
    for profile_key in profile_keys:
        profiles.pop(profile_key)

    with PROFILES_FILE.open("w", encoding="utf-8") as profiles_file:
        json.dump(profiles, profiles_file, indent=4)
        profiles_file.write("\n")

    for profile_key in profile_keys:
        if re.fullmatch(r"face_\d+", profile_key):
            crop_path = CROPS_DIR / f"{profile_key}.jpg"
            if crop_path.exists():
                crop_path.unlink()


def remove_legacy_profile(student_id: str) -> None:
    """Remove an identity from the legacy single-sample gallery when present."""
    if not LEGACY_GALLERY_FILE.exists():
        return

    with LEGACY_GALLERY_FILE.open("rb") as gallery_file:
        legacy_gallery = pickle.load(gallery_file)
    if not isinstance(legacy_gallery, list):
        return

    filtered_gallery = [
        profile
        for profile in legacy_gallery
        if str(profile.get("student_id")) != student_id
    ]
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=LEGACY_GALLERY_FILE.parent, delete=False, suffix=".pkl"
    ) as temporary_file:
        pickle.dump(filtered_gallery, temporary_file)
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(LEGACY_GALLERY_FILE)


def delete_profile(student_id: str, confirmed: bool) -> tuple[list[list[str]], str]:
    """Delete one profile and its related project metadata after explicit confirmation."""
    student_id = student_id.strip()
    if not student_id:
        return profile_rows(), "Enter the student ID to delete."
    if not confirmed:
        return profile_rows(), "Tick the confirmation box before deleting a profile."

    gallery = load_gallery()
    gallery_key = next((key for key in gallery if str(key) == student_id), None)
    if gallery_key is None:
        return profile_rows(), f"No profile exists with student ID {student_id}."

    removed_profile = gallery[gallery_key]
    gallery_to_save = {
        str(key): {**profile, "embeddings": list(profile["embeddings"])}
        for key, profile in gallery.items()
        if key != gallery_key
    }
    try:
        save_gallery(gallery_to_save)
        remove_profile_registry(student_id)
        remove_legacy_profile(student_id)
    except Exception as error:
        LOGGER.exception("Could not delete profile %s", student_id)
        return profile_rows(), f"The profile could not be deleted completely: {error}"

    return (
        profile_rows(),
        f"Deleted **{removed_profile['name']} ({student_id})** and its saved face samples."
    )


def clear_form() -> tuple[None, None, list[tuple[np.ndarray, str]], list[list[str]], list[dict[str, Any]], list[list[str]], str]:
    """Clear the workflow and refresh the saved-profile table."""
    return None, None, [], [], [], profile_rows(), "Ready to detect faces in a group photo."


def create_app() -> gr.Blocks:
    """Create the multi-face registration interface without launching a server."""
    with gr.Blocks(title="Group Face Profile Registration") as interface:
        gr.Markdown(
            "# Group Face Profile Registration\n"
            "Upload one group photo. Known students are filtered out, leaving only unregistered faces for you to name and save."
        )
        detected_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                uploader = gr.File(
                    label="Group photo (JPG, PNG, HEIC, or HEIF)",
                    file_types=SUPPORTED_IMAGE_TYPES,
                    type="filepath",
                )
                with gr.Row():
                    detect_button = gr.Button("Detect Faces", variant="primary")
                    save_button = gr.Button("Save Named Faces", variant="primary")
                    clear_button = gr.Button("Clear")
                status = gr.Markdown("Ready to detect faces in a group photo.")
            with gr.Column(scale=1):
                annotated_image = gr.Image(label="Detected faces", type="numpy")

        gr.Markdown("## Unregistered face assignments")
        gr.Markdown("Only faces not confidently matched to a saved profile appear here. Fill in **Student ID** and **Name** for each one, or leave a row blank to skip it.")
        assignment_table = gr.Dataframe(
            headers=["Detected face", "Student ID", "Name"],
            datatype=["str", "str", "str"],
            interactive=True,
            label="Unregistered face details",
        )
        face_gallery = gr.Gallery(label="Unregistered face crops", columns=4, object_fit="cover")

        gr.Markdown("## All available face profiles")
        saved_profiles = gr.Dataframe(
            value=profile_rows(),
            headers=["Student ID", "Name", "Samples"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="Live multi-sample gallery",
        )

        gr.Markdown("## Edit an existing profile")
        gr.Markdown("Changing an ID or name preserves that student's saved face samples.")
        with gr.Row():
            current_id_input = gr.Textbox(label="Current student ID")
            updated_id_input = gr.Textbox(label="New student ID")
            updated_name_input = gr.Textbox(label="New name")
            edit_button = gr.Button("Update Profile")

        gr.Markdown("## Delete a face profile")
        gr.Markdown("Deleting removes the profile, its saved embeddings, and its representative crop. This cannot be undone from the app.")
        with gr.Row():
            delete_id_input = gr.Textbox(label="Student ID to delete")
            delete_confirm = gr.Checkbox(label="I confirm that I want to permanently delete this profile")
            delete_button = gr.Button("Delete Profile", variant="stop")

        detect_button.click(
            detect_group_faces,
            inputs=uploader,
            outputs=[annotated_image, face_gallery, assignment_table, detected_state, status],
        )
        save_button.click(
            save_assignments,
            inputs=[detected_state, assignment_table],
            outputs=[saved_profiles, status],
        )
        edit_button.click(
            edit_profile,
            inputs=[current_id_input, updated_id_input, updated_name_input],
            outputs=[saved_profiles, status],
        )
        delete_button.click(
            delete_profile,
            inputs=[delete_id_input, delete_confirm],
            outputs=[saved_profiles, status],
        )
        clear_button.click(
            clear_form,
            outputs=[uploader, annotated_image, face_gallery, assignment_table, detected_state, saved_profiles, status],
        )

    return interface


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    create_app().launch()
