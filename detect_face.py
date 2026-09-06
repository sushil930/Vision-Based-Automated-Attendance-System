import os
import cv2
from insightface.app import FaceAnalysis


INPUT_DIR = "data/test"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded!\n")


image_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )
])


if not image_files:
    print("No images found in:", INPUT_DIR)
    exit()


total_faces = 0


for filename in image_files:

    input_path = os.path.join(
        INPUT_DIR,
        filename
    )

    print("=" * 60)
    print(f"Processing: {filename}")

    image = cv2.imread(input_path)

    if image is None:
        print("Could not read image!")
        continue

    height, width = image.shape[:2]

    print(f"Image size: {width} x {height}")

    faces = app.get(image)

    print(f"Faces detected: {len(faces)}")

    total_faces += len(faces)

    for i, face in enumerate(faces):

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        confidence = float(face.det_score)

        print(
            f"Face {i + 1}: "
            f"confidence={confidence:.3f}"
        )

        # Draw bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # Label
        label = (
            f"Face {i + 1} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"detected_{filename}"
    )

    cv2.imwrite(
        output_path,
        image
    )

    print(f"Saved → {output_path}")


print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)

print(f"Images processed: {len(image_files)}")
print(f"Total faces detected: {total_faces}")

if image_files:
    print(
        f"Average faces/image: "
        f"{total_faces / len(image_files):.2f}"
    )

print("=" * 60)