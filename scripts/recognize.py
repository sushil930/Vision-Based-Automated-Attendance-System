import cv2
import pickle
import numpy as np

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from insightface.app import FaceAnalysis


IMAGE_PATH = "data/test/test05.jpg"

GALLERY_FILE = (
    "gallery/multi_sample_gallery.pkl"
)

OUTPUT_IMAGE = (
    "results/multi_sample_result.jpg"
)


# Recognition thresholds
MATCH_THRESHOLD = 0.55
REVIEW_THRESHOLD = 0.45


register_heif_opener()


# -----------------------------------------
# Load model
# -----------------------------------------

print("Loading model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)


# -----------------------------------------
# Load gallery
# -----------------------------------------

with open(
    GALLERY_FILE,
    "rb"
) as f:

    gallery = pickle.load(f)


# -----------------------------------------
# Similarity
# -----------------------------------------

def cosine_similarity(a, b):

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    return float(
        np.dot(a, b)
    )


# -----------------------------------------
# Load test image
# -----------------------------------------

def load_image(image_path: str) -> np.ndarray | None:
    """Load a standard image or HEIC/HEIF image in OpenCV BGR format."""

    if image_path.lower().endswith((".heic", ".heif")):
        with Image.open(image_path) as pil_image:
            rgb_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            return cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)

    return cv2.imread(image_path)


image = load_image(IMAGE_PATH)

if image is None:

    raise FileNotFoundError(
        IMAGE_PATH
    )


faces = app.get(image)


print()
print(
    f"Faces detected: {len(faces)}"
)


recognized = {}


# -----------------------------------------
# Recognize each face
# -----------------------------------------

for index, face in enumerate(
    faces,
    start=1
):

    embedding = face.embedding.astype(
        np.float32
    )

    embedding /= np.linalg.norm(
        embedding
    )


    person_scores = []


    # -------------------------------------
    # Compare against each person
    # -------------------------------------

    for student_id, profile in gallery.items():

        best_person_score = -1


        for stored_embedding in profile[
            "embeddings"
        ]:

            score = cosine_similarity(
                embedding,
                stored_embedding
            )

            best_person_score = max(
                best_person_score,
                score
            )


        person_scores.append({
            "student_id": student_id,
            "name": profile["name"],
            "score": best_person_score
        })


    # Sort best match first
    person_scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    best = person_scores[0]

    second = person_scores[1] if len(person_scores) > 1 else None


    # -------------------------------------
    # Decision
    # -------------------------------------

    margin = (
        best["score"] -
        second["score"]
        if second
        else 1.0
    )


    if (
        best["score"] >= MATCH_THRESHOLD
        and margin >= 0.05
    ):

        status = "MATCH"

        name = best["name"]

        recognized[
            best["student_id"]
        ] = {
            "name": name,
            "score": best["score"]
        }


    elif best["score"] >= REVIEW_THRESHOLD:

        status = "REVIEW"

        name = (
            "?"
            + best["name"]
        )


    else:

        status = "UNKNOWN"

        name = "UNKNOWN"


    print(
        f"Face {index}: "
        f"{name} | "
        f"score={best['score']:.3f} | "
        f"margin={margin:.3f} | "
        f"{status}"
    )


    # -------------------------------------
    # Draw
    # -------------------------------------

    x1, y1, x2, y2 = (
        face.bbox.astype(int)
    )


    if status == "MATCH":

        color = (0, 255, 0)

    elif status == "REVIEW":

        color = (0, 255, 255)

    else:

        color = (0, 0, 255)


    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        3
    )


    label = (
        f"{name} "
        f"{best['score']:.2f}"
    )


    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


# -----------------------------------------
# Save
# -----------------------------------------

cv2.imwrite(
    OUTPUT_IMAGE,
    image
)


print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(
    f"Unique recognized: "
    f"{len(recognized)}"
)

for student_id, result in recognized.items():

    print(
        f"{student_id} | "
        f"{result['name']} | "
        f"{result['score']:.3f}"
    )

print()
print(
    f"Saved → {OUTPUT_IMAGE}"
)
