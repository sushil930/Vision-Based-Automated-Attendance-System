"""Add face samples from enrollment images to the multi-sample gallery.

Usage:
    python scripts/add_samples.py [image_directory]

Processes every image in the directory, detects faces, matches them against
known gallery profiles, and adds high-confidence matches as additional
samples to improve recognition accuracy.

Default image directory: data/enrollment
"""

import sys
import pickle
import tempfile
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GALLERY_FILE = PROJECT_ROOT / "gallery" / "multi_sample_gallery.pkl"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "enrollment"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Only auto-add when the match is very confident.
ADD_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.55


def load_gallery() -> dict:
    """Load the multi-sample gallery from disk."""
    with GALLERY_FILE.open("rb") as f:
        gallery = pickle.load(f)
    if not isinstance(gallery, dict) or not gallery:
        raise ValueError(f"Gallery is empty or invalid: {GALLERY_FILE}")
    return gallery


def save_gallery(gallery: dict) -> None:
    """Atomically replace the gallery file."""
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=GALLERY_FILE.parent, delete=False, suffix=".pkl"
    ) as tmp:
        pickle.dump(gallery, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(GALLERY_FILE)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def find_best_match(
    embedding: np.ndarray, gallery: dict
) -> tuple[str | None, str | None, float]:
    """Return (student_id, name, best_score) for the closest gallery profile."""
    best_id = None
    best_name = None
    best_score = -1.0

    for student_id, profile in gallery.items():
        for stored in profile["embeddings"]:
            score = cosine_similarity(
                embedding, np.asarray(stored, dtype=np.float32)
            )
            if score > best_score:
                best_score = score
                best_id = str(student_id)
                best_name = profile["name"]

    return best_id, best_name, best_score


def main() -> None:
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_DIR
    if not image_dir.is_dir():
        print(f"Directory not found: {image_dir}")
        sys.exit(1)

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        print(f"No images found in: {image_dir}")
        sys.exit(1)

    print("Loading InsightFace...")
    model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    model.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded.\n")

    gallery = load_gallery()
    print(f"Gallery profiles: {len(gallery)}")
    print(f"Images to process: {len(image_files)}\n")

    added = 0
    reviewed = 0
    unknown = 0

    for image_path in image_files:
        print("=" * 60)
        print(f"Processing: {image_path.name}")

        image = cv2.imread(str(image_path))
        if image is None:
            print("  Could not read image, skipping.")
            continue

        faces = model.get(image)
        print(f"  Faces detected: {len(faces)}")

        for i, face in enumerate(faces, start=1):
            embedding = face.embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print(f"  Face {i}: invalid embedding, skipping.")
                continue
            embedding /= norm

            match_id, match_name, score = find_best_match(embedding, gallery)

            if score >= ADD_THRESHOLD:
                gallery[match_id]["embeddings"].append(embedding)
                added += 1
                print(f"  Face {i}: {match_name} ({score:.3f}) [ADDED]")
            elif score >= REVIEW_THRESHOLD:
                reviewed += 1
                print(
                    f"  Face {i}: {match_name} ({score:.3f}) [REVIEW - not added]"
                )
            else:
                unknown += 1
                print(f"  Face {i}: Unknown ({score:.3f})")

    save_gallery(gallery)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Samples added:    {added}")
    print(f"Skipped (review): {reviewed}")
    print(f"Unknown faces:    {unknown}")
    print()

    for student_id, profile in sorted(
        gallery.items(), key=lambda x: str(x[0])
    ):
        print(
            f"  {student_id} | {profile['name']} "
            f"| samples={len(profile['embeddings'])}"
        )

    print(f"\nSaved → {GALLERY_FILE}")


if __name__ == "__main__":
    main()
