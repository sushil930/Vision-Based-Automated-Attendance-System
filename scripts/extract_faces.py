import os
import cv2
from insightface.app import FaceAnalysis

# Updated to match the actual file name found: group1.jpg
INPUT_IMAGE = "data/enroll/group1.jpg"

OUTPUT_DIR = "gallery/crops"
OUTPUT_IMAGE = "results/enrollment_faces.jpg"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True)

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

image = cv2.imread(INPUT_IMAGE)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {INPUT_IMAGE}"
    )

faces = app.get(image)

print(f"\nDetected faces: {len(faces)}")

# Sort faces roughly left-to-right, then top-to-bottom.
faces = sorted(
    faces,
    key=lambda face: (
        face.bbox[1],
        face.bbox[0]
    )
)

for index, face in enumerate(faces, start=1):
    x1, y1, x2, y2 = face.bbox.astype(int)

    # Add padding around face
    width = x2 - x1
    height = y2 - y1

    padding_x = int(width * 0.25)
    padding_y = int(height * 0.35)

    crop_x1 = max(0, x1 - padding_x)
    crop_y1 = max(0, y1 - padding_y)
    crop_x2 = min(image.shape[1], x2 + padding_x)
    crop_y2 = min(image.shape[0], y2 + padding_y)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_path = os.path.join(OUTPUT_DIR, f"face_{index:02d}.jpg")
    cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Draw on original image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"FACE {index}"
    cv2.putText(image, label, (x1, max(y1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    print(f"Face {index} -> {crop_path}")

cv2.imwrite(OUTPUT_IMAGE, image)

print("\nDone!")
print(f"Annotated image: {OUTPUT_IMAGE}")
print(f"Face crops: {OUTPUT_DIR}/")
