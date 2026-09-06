import os
import json
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

CROPS_DIR = "gallery/crops"
PROFILES_FILE = "gallery/profiles.json"
OUTPUT_FILE = "gallery/face_gallery.pkl"

print("Loading InsightFace...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded.")

with open(PROFILES_FILE, "r") as f:
    profiles = json.load(f)

gallery = []

for crop_name, profile in profiles.items():
    filename = f"{crop_name}.jpg"
    image_path = os.path.join(CROPS_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        continue

    # First attempt at detection
    faces = app.get(image)

    # Fallback: If no face found, try extremely low threshold for crops
    if len(faces) == 0:
        app.models['detection'].det_thresh = 0.1
        faces = app.get(image)
        app.models['detection'].det_thresh = 0.5  # Reset to default

    if len(faces) == 0:
        print(f"WARNING: No face found in {filename} even with 0.1 threshold. Skipping.")
        continue

    # Choose largest detected face
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    
    embedding = face.embedding.astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    gallery.append({
        "student_id": profile["student_id"],
        "name": profile["name"],
        "embedding": embedding
    })
    print(f"Added: {profile['student_id']} - {profile['name']}")

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(gallery, f)

print(f"\nDONE! Profiles in gallery: {len(gallery)}")
print(f"Saved: {OUTPUT_FILE}")
