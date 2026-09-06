
import os
import cv2
import pickle

import numpy as np

from insightface.app import FaceAnalysis


INPUT_IMAGE = "data/test/group2.jpg"

GALLERY_FILE = "gallery/face_gallery.pkl"

OUTPUT_IMAGE = "results/recognized_group2.jpg"

# Initial threshold only.
# We will calibrate this later.
THRESHOLD = 0.45


# --------------------------------
# Load model
# --------------------------------

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Model loaded.")


# --------------------------------
# Load gallery
# --------------------------------

with open(GALLERY_FILE, "rb") as f:
    gallery = pickle.load(f)


print(
    f"Gallery profiles: {len(gallery)}"
)


# --------------------------------
# Load image
# --------------------------------

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError(
        INPUT_IMAGE
    )


# --------------------------------
# Detect faces
# --------------------------------

faces = app.get(image)

print(
    f"\nFaces detected: {len(faces)}"
)


# --------------------------------
# Cosine similarity
# --------------------------------

def cosine_similarity(a, b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return float(
        np.dot(a, b)
    )


recognized = {}


# --------------------------------
# Process faces
# --------------------------------

for index, face in enumerate(
    faces,
    start=1
):

    embedding = face.embedding.astype(
        np.float32
    )

    embedding = embedding / np.linalg.norm(
        embedding
    )

    best_match = None
    best_score = -1


    # Compare against every profile
    for profile in gallery:

        score = cosine_similarity(
            embedding,
            profile["embedding"]
        )

        if score > best_score:

            best_score = score
            best_match = profile


    x1, y1, x2, y2 = (
        face.bbox.astype(int)
    )


    if best_score >= THRESHOLD:

        name = best_match["name"]
        student_id = best_match["student_id"]

        status = "MATCH"

        recognized[
            student_id
        ] = {
            "name": name,
            "confidence": best_score
        }

    else:

        name = "UNKNOWN"
        student_id = None

        status = "UNKNOWN"


    print(
        f"Face {index}: "
        f"{name} | "
        f"similarity={best_score:.3f} | "
        f"{status}"
    )


    # --------------------------------
    # Draw result
    # --------------------------------

    if status == "MATCH":

        box_color = (0, 255, 0)

    else:

        box_color = (0, 0, 255)


    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        box_color,
        3
    )


    label = (
        f"{name} "
        f"{best_score:.2f}"
    )


    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        box_color,
        2,
        cv2.LINE_AA
    )


# --------------------------------
# Save result
# --------------------------------

cv2.imwrite(
    OUTPUT_IMAGE,
    image
)


print()
print("=" * 60)
print("RECOGNITION RESULT")
print("=" * 60)

print(
    f"Faces detected: {len(faces)}"
)

print(
    f"Unique people recognized: "
    f"{len(recognized)}"
)

print()

for student_id, result in recognized.items():

    print(
        f"{student_id} | "
        f"{result['name']} | "
        f"{result['confidence']:.3f}"
    )


print()
print(
    f"Result saved: {OUTPUT_IMAGE}"
)
