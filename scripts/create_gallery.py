import os
import cv2
import pickle

import numpy as np

from insightface.app import FaceAnalysis


# -----------------------------------------
# Configuration
# -----------------------------------------

IMAGE_DIR = "data/enrollment"

OLD_GALLERY = "gallery/face_gallery.pkl"

NEW_GALLERY = "gallery/multi_sample_gallery.pkl"

THRESHOLD = 0.55


# -----------------------------------------
# Load InsightFace
# -----------------------------------------

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


# -----------------------------------------
# Load existing gallery
# -----------------------------------------

with open(OLD_GALLERY, "rb") as f:
    old_gallery = pickle.load(f)


print(
    f"Existing profiles: {len(old_gallery)}"
)


# -----------------------------------------
# Create multi-sample gallery
# -----------------------------------------

gallery = {}

for profile in old_gallery:

    student_id = profile["student_id"]

    gallery[student_id] = {
        "student_id": student_id,
        "name": profile["name"],
        "embeddings": [
            profile["embedding"]
        ]
    }


# -----------------------------------------
# Cosine similarity
# -----------------------------------------

def cosine_similarity(a, b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return float(
        np.dot(a, b)
    )


# -----------------------------------------
# Process enrollment images
# -----------------------------------------

image_files = sorted([
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
])


print(
    f"\nImages found: {len(image_files)}"
)


for filename in image_files:

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    print()
    print("=" * 60)
    print(f"Processing: {filename}")

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        continue

    faces = app.get(image)

    print(
        f"Faces detected: {len(faces)}"
    )


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


        # --------------------------------
        # Find closest known person
        # --------------------------------

        best_id = None
        best_name = None
        best_score = -1


        for student_id, profile in gallery.items():

            for stored_embedding in profile[
                "embeddings"
            ]:

                score = cosine_similarity(
                    embedding,
                    stored_embedding
                )

                if score > best_score:

                    best_score = score
                    best_id = student_id
                    best_name = profile["name"]


        # --------------------------------
        # Add sample if confident
        # --------------------------------

        if best_score >= 0.70:

            gallery[best_id]["embeddings"].append(embedding)

            print(
                f"Face {index}: "
                f"{best_name} "
                f"{best_score:.3f} "
                f"[ADDED]"
            )

        elif best_score >= 0.55:

            print(
                f"Face {index}: "
                f"{best_name} "
                f"{best_score:.3f} "
                f"[REVIEW]"
            )

        else:

            print(
                f"Face {index}: "
                f"UNKNOWN "
                f"{best_score:.3f}"
            )


# -----------------------------------------
# Save
# -----------------------------------------

with open(NEW_GALLERY, "wb") as f:

    pickle.dump(
        gallery,
        f
    )


print()
print("=" * 60)
print("MULTI-SAMPLE GALLERY CREATED")
print("=" * 60)

for student_id, profile in gallery.items():

    print(
        f"{student_id} | "
        f"{profile['name']} | "
        f"samples={len(profile['embeddings'])}"
    )

print()
print(
    f"Saved → {NEW_GALLERY}"
)
