# Fully Offline Mobile AI Classroom Attendance App
## Full Implementation Plan & Coding-Agent Instructions

> **Project goal:** Build an Android-first Flutter application that performs classroom face detection, face recognition, attendance, enrollment, review, and local storage completely offline, with a target release size of **under 100 MB**.
>
> **Critical constraint:** The mobile app must not depend on the existing Python/FastAPI server for normal attendance. The backend can remain as an optional future synchronization service.

---

## 1. Product Goal

Build a teacher-facing mobile application that can:

1. Create and manage classes and subjects.
2. Register students.
3. Enroll student faces completely offline.
4. Capture one or two classroom/group photos.
5. Detect all visible faces locally.
6. Recognize enrolled students locally.
7. Mark high-confidence recognized students as `PRESENT` candidates.
8. Send uncertain cases to `REVIEW`.
9. Keep unrecognized faces as `UNKNOWN`.
10. Calculate absent students from the registered class roster.
11. Let the teacher manually correct attendance.
12. Save attendance locally in SQLite.
13. Work with Wi-Fi/mobile data disabled.
14. Optionally synchronize later when connectivity exists.
15. Target a final release package under **100 MB**.

The system must optimize for privacy, predictable behavior, low latency, and graceful failure.

---

# 2. Non-Negotiable Requirements

## Offline-first

These must work with airplane mode enabled:

- App launch after installation
- Local teacher profile/session
- Class creation
- Student registration
- Subject management
- Face enrollment
- Camera capture
- Gallery import
- Face detection
- Face recognition
- Attendance calculation
- Manual review/correction
- Attendance history
- Local export/share
- Local backup/restore

The recognition and attendance paths must contain **no required network request** and must not silently fall back to a server.

## Package size

Target:

```text
Release artifact < 100 MB
```

Prefer:

```text
App + runtime + ML models <= 80 MB
```

Leave headroom for future releases.

Do **not** ship:

- Python runtime
- InsightFace Python package
- `buffalo_l` model pack
- `buffalo_s` model pack
- training datasets
- test datasets
- unnecessary sample images
- development/debug assets
- unused native architectures in direct APKs

The current InsightFace model zoo documents `buffalo_l` at about 326 MB and `buffalo_s` at about 159 MB as model packs, which is incompatible with the project's size target. It also documents that those packs contain different recognition models, so their embedding spaces must not be mixed. Source: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md

---

# 3. Recommended High-Level Architecture

```text
Flutter UI
    |
    +-- Camera / Image Picker
    |
    v
Image Preprocessor
    |
    v
Lightweight Face Detector
    |
    v
Face Quality Check
    |
    v
Face Alignment / Crop
    |
    v
Lightweight Face Recognition Encoder
    |
    v
Normalized Embedding
    |
    v
Local Gallery Index
    |
    v
Cosine Similarity / Vector Search
    |
    v
MATCH / REVIEW / UNKNOWN
    |
    v
Attendance Engine
    |
    +--> SQLite
    +--> Review UI
    +--> History / Export
    +--> Optional future Sync
```

The production attendance path must stay on-device.

---

# 4. Technology Stack

## Mobile framework

Use:

```text
Flutter
Dart
Android-first
```

Use Flutter's official Android release/build guidance for release packaging, app bundles, ABI splitting, and code shrinking:
https://docs.flutter.dev/deployment/android

## Local database

Use:

```text
SQLite
```

Keep database access behind repositories so the storage implementation is isolated from the rest of the app.

## ML runtime

Evaluate both:

```text
LiteRT / TensorFlow Lite
```

and:

```text
ONNX Runtime Mobile
```

Do not select the runtime only because its Flutter integration is convenient. Benchmark:

- model size
- cold initialization time
- warm inference time
- peak RAM
- Android compatibility
- native dependency size
- delegate support
- integration complexity

ONNX Runtime officially supports mobile inference on Android/iOS and notes that models must fit device storage and memory. It also provides mobile-specific runtime/model optimization paths:
https://onnxruntime.ai/docs/tutorials/mobile/
https://onnxruntime.ai/docs/get-started/with-mobile.html

If ONNX Runtime is selected, investigate reduced/custom operator builds before shipping the full runtime.

## Face detector

Candidate choices:

```text
MediaPipe Face Detector
lightweight SCRFD/mobile detector
BlazeFace or similar compact detector
other properly licensed mobile detector
```

MediaPipe documents a model-based Face Detector that supports image/video/live-stream modes and returns face detections/keypoints:
https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceDetector

Do not select a detector solely by model size. Benchmark classroom detection recall.

## Face recognition model

Evaluate lightweight mobile encoders such as:

```text
MobileFaceNet
lightweight ArcFace-compatible mobile encoder
another compact, properly licensed face-recognition model
```

The final model must:

- be licensed for the intended use
- be distributable with the app
- run on Android devices
- support the chosen ML runtime
- generate stable embeddings
- support cosine similarity or equivalent matching
- perform acceptably on the project's test population
- remain within the size budget

Do not assume a model is suitable because its name contains `Mobile`, `FaceNet`, or `ArcFace`.

---

# 5. Critical Model Compatibility Rule

The desktop system currently uses:

```text
InsightFace buffalo_l
```

The mobile app will use a different mobile encoder.

Therefore the mobile gallery must be generated with the **same exact mobile model used during attendance**.

```text
Mobile enrollment model
        ==
Mobile attendance model
```

The following must also remain identical:

- preprocessing
- face alignment
- image normalization
- color channel order
- input dimensions
- embedding normalization
- model version

Never compare embeddings generated by different recognition models.

---

# 6. Model Selection Phase

Before building the complete app UI, create a standalone Android ML benchmark.

## Detector benchmark

Test multiple viable candidates.

Measure:

- packaged size
- startup time
- warm inference time
- RAM
- detected faces
- missed faces
- false detections
- small faces
- side faces
- partial occlusion
- poor lighting
- classroom-distance faces

## Recognition benchmark

For each candidate encoder:

1. Enroll the same people using the candidate model.
2. Generate fresh embeddings.
3. Test unseen images.
4. Test group images.
5. Test unknown people.
6. Calculate genuine similarity scores.
7. Calculate impostor similarity scores.
8. Measure false accepts/rejects.
9. Measure inference latency.
10. Record model/package size.

Do not reuse desktop thresholds.

---

# 7. Project Structure

Use a clean separation between UI, domain logic, local data, and ML.

```text
mobile_app/
├── android/
├── assets/
│   ├── models/
│   │   ├── face_detector.*
│   │   └── face_recognizer.*
│   └── icons/
│
├── lib/
│   ├── main.dart
│   ├── app/
│   │   ├── app.dart
│   │   ├── router.dart
│   │   └── theme.dart
│   ├── core/
│   │   ├── constants/
│   │   ├── errors/
│   │   ├── logging/
│   │   ├── permissions/
│   │   └── utils/
│   ├── data/
│   │   ├── db/
│   │   │   ├── database.dart
│   │   │   └── migrations/
│   │   ├── models/
│   │   ├── repositories/
│   │   └── local/
│   ├── domain/
│   │   ├── entities/
│   │   ├── services/
│   │   └── use_cases/
│   ├── features/
│   │   ├── onboarding/
│   │   ├── dashboard/
│   │   ├── classes/
│   │   ├── students/
│   │   ├── enrollment/
│   │   ├── attendance/
│   │   ├── review/
│   │   ├── history/
│   │   └── settings/
│   └── ml/
│       ├── detector/
│       ├── recognizer/
│       ├── preprocessing/
│       ├── embeddings/
│       ├── matcher/
│       ├── quality/
│       └── benchmark/
├── test/
├── integration_test/
└── pubspec.yaml
```

Do not place ML inference logic directly inside screen widgets.

---

# 8. Core Data Model

Use SQLite.

## teachers

```text
id
name
created_at
updated_at
```

## classes

```text
id
name
section
semester
academic_year
created_at
updated_at
```

## subjects

```text
id
name
code
created_at
updated_at
```

## class_subjects

```text
id
class_id
subject_id
```

## students

```text
id
class_id
roll_number
name
status
created_at
updated_at
```

## face_profiles

```text
id
student_id
model_id
model_version
embedding_dimension
preprocessing_version
created_at
updated_at
```

## face_embeddings

```text
id
face_profile_id
embedding_blob
sample_index
quality_score
created_at
```

Store normalized embeddings efficiently.

Do not store full-resolution enrollment images by default.

## attendance_sessions

```text
id
class_id
subject_id
date
started_at
completed_at
status
```

## attendance_records

```text
id
session_id
student_id
status
confidence
source_image_count
manually_modified
review_status
updated_at
```

Possible status values:

```text
PRESENT
ABSENT
REVIEW
```

## recognition_events (optional)

```text
id
session_id
student_id nullable
score
margin
decision
face_index
created_at
```

Do not persist raw face images or embeddings in logs.

---

# 9. Enrollment Workflow

Enrollment is the foundation of recognition quality.

## Student registration

Teacher enters:

```text
Name
Roll Number
Class
```

Then chooses:

```text
Enroll Face
```

## Capture 3–5 samples

Use varied but reasonable poses:

```text
front
slightly left
slightly right
normal expression
normal indoor lighting
```

Avoid:

```text
blur
extreme pose
very dark images
multiple faces
very small face
heavy obstruction
```

## Quality checks

Ask for a retake when:

- no face is found
- more than one face is found
- face is too small
- image is severely blurred
- pose is extreme
- exposure is unusable
- face is heavily occluded

## Embedding generation

```text
capture
 ↓
detect
 ↓
align
 ↓
recognition encoder
 ↓
embedding
 ↓
L2 normalize
 ↓
store
```

## Sample strategy

Start by storing 3–5 normalized embeddings per student.

Do not blindly average every sample during v1.

At recognition:

```text
query embedding
    ↓
compare to each sample
    ↓
best score for each student
    ↓
best student + margin
```

Later benchmark whether prototype averaging improves performance.

---

# 10. Recognition Pipeline

```text
Input image
   ↓
EXIF/orientation correction
   ↓
large-image downscale
   ↓
face detection
   ↓
face quality filtering
   ↓
face alignment
   ↓
embedding inference
   ↓
L2 normalization
   ↓
local vector matching
   ↓
score + margin
   ↓
MATCH / REVIEW / UNKNOWN
```

All preprocessing used for enrollment must be reused for attendance.

---

# 11. Face Detection Rules

The detector must support multiple faces per image.

Do not:

- stop after the first face
- assume one face per photo
- recognize the whole image as a single identity

Use configurable:

```text
min_detection_confidence
min_face_size
max_faces
```

Do not pick final values without benchmark data.

Tiny faces may be detected but marked unusable:

```text
face detected
     ↓
quality too low
     ↓
REVIEW / MISSED
```

Do not spend expensive recognition inference on hopeless crops.

---

# 12. Image Preprocessing

Centralize all preprocessing in one service.

Required stages:

```text
orientation correction
    ↓
resize/downscale
    ↓
color conversion
    ↓
face crop
    ↓
alignment
    ↓
model normalization
```

Never duplicate preprocessing code between enrollment and attendance.

---

# 13. Image Size Strategy

Start with:

```text
MAX_IMAGE_SIZE = 1280
```

Rules:

```text
longest edge <= MAX_IMAGE_SIZE
    → keep original

longest edge > MAX_IMAGE_SIZE
    → downscale proportionally
```

Do not upscale.

Benchmark:

```text
960
1280
1600
1920
```

Select the lowest setting that preserves classroom face recall.

---

# 14. Local Gallery Matcher

Follow the proven desktop approach conceptually.

Store:

```text
gallery_matrix
student_ids
starts
embedding_dimension
model_id
model_version
```

All embeddings should be normalized once.

Recognition should use vectorized operations where the selected runtime/platform allows it.

Conceptually:

```text
query_embeddings @ gallery_matrix.T
```

Then calculate:

```text
best sample per student
best score
second-best score
margin
```

Avoid nested Dart loops when a native/vectorized path is available.

Do not add FAISS/vector databases until measurements show they are needed.

---

# 15. Recognition Decision Logic

Use three states:

```text
MATCH
REVIEW
UNKNOWN
```

Generic structure:

```text
if score >= MATCH_THRESHOLD
and margin >= MIN_MARGIN:
    MATCH

else if score >= REVIEW_THRESHOLD:
    REVIEW

else:
    UNKNOWN
```

Thresholds must be calibrated using the mobile model's own validation data.

Never copy thresholds from `buffalo_l`.

---

# 16. Unknown Person Handling

This is mandatory.

Create test cases containing:

```text
registered students
+
unregistered people
```

A nearest candidate is not automatically a valid identity.

Example:

```text
closest student = A
score = moderate
margin = low
```

Result:

```text
REVIEW
```

or:

```text
UNKNOWN
```

based on calibrated thresholds.

Never automatically mark an unknown person as present.

---

# 17. Two-Photo Attendance

Support:

```text
Photo 1
Photo 2
```

Use the same model instances for both images.

For every result store:

```text
student_id
best_score
best_margin
source_photo
```

Merge by student ID:

```text
same student in both photos
        ↓
keep strongest valid evidence
```

Then:

```text
recognized = union(photo1_ids, photo2_ids)
```

Attendance calculation:

```text
confirmed_present = recognized MATCHes + teacher-confirmed REVIEWs
absent = registered - confirmed_present
```

Unresolved `REVIEW` must remain unresolved.

---

# 18. Attendance Review UI

For a review item show:

```text
Student candidate: Rahul Sharma
Confidence: 0.61
Status: REVIEW
Face crop: [image]

[Confirm Present]
[Mark Absent]
[Unknown]
```

For unknown faces:

```text
Unknown face
[Assign student]
[Ignore]
```

Make uncertainty visible and easy to resolve.

---

# 19. Main Screens

## Splash / startup

Show:

```text
Loading offline AI...
```

Initialize:

- database
- model manager
- detector
- recognizer
- gallery index
- configuration

Do not repeatedly initialize models.

## Onboarding

Explain:

```text
Works offline
Face processing happens on this device
Internet is optional
```

## Dashboard

Show:

```text
Today's classes
Recent attendance
Quick actions
```

## Classes

Actions:

```text
Create class
Open class
Manage students
Manage subjects
```

## Student list

Show:

```text
Roll number
Name
Face enrolled / not enrolled
```

## Enrollment

Show:

```text
Student details
Capture progress
Quality feedback
Embedding progress
Enrollment result
```

## Attendance

Flow:

```text
Select class
Select subject
Start attendance
Capture photo
Optional second photo
Process
Review
Confirm
Save
```

## Results

Show:

```text
Present count
Absent count
Review count
Unknown count
```

## History

Filter by:

```text
class
subject
date
```

## Settings

Include:

```text
Model information
Storage usage
Export
Backup/restore
Clear face data
Clear attendance data
Privacy
About
```

Threshold controls should be developer/debug only unless there is a deliberate product requirement.

---

# 20. Camera UX

Attendance is photo-based in v1.

Camera flow:

```text
Preview
 ↓
Capture
 ↓
Review photo
 ↓
Use / Retake
 ↓
Process
```

Do not run full recognition on every preview frame.

For future live-assistance mode, use a lightweight detector/tracker and recognize only suitable frames.

---

# 21. Background Processing

ML inference must not freeze the Flutter UI.

Use an appropriate background strategy:

```text
Dart isolate
native worker thread
runtime-specific async/background API
```

The exact mechanism depends on the selected ML package.

Progress states should include:

```text
Preparing image...
Finding faces...
Recognizing faces...
Calculating attendance...
```

---

# 22. Batch Recognition Strategy

For a group image:

```text
detect all faces
    ↓
crop faces
    ↓
preprocess faces
    ↓
run embedding inference
    ↓
vectorized matching
```

Use batch inference if the selected mobile model/runtime supports it.

If the model only supports batch size 1, still batch surrounding preprocessing and local matching where possible.

---

# 23. Performance Targets

Use targets as engineering goals, not guarantees.

Initial goals:

```text
Model warmup: one-time only
Single classroom photo: < 2–3 seconds target on mid-range hardware
20-face photo: < 2–3 seconds target where hardware permits
Gallery matching: < 50 ms target
UI: no visible blocking
```

Benchmark at least:

```text
low-end Android
mid-range Android
high-end Android
```

Record:

```text
startup time
model load time
detection time
embedding time
matching time
annotation time
total latency
peak RAM
battery impact
thermal throttling
```

---

# 24. Performance Optimization Order

Optimize in this order:

```text
1. Face detector
2. Recognition model inference
3. Input image dimensions
4. Image copies/preprocessing
5. Model initialization
6. Memory allocations
7. Gallery matching
8. UI rendering
```

Do not prematurely optimize the gallery.

The desktop benchmark already showed that gallery matching is tiny compared with face detection.

---

# 25. Model Warm-Up

Load detector/recognizer once.

Warm them once after initialization if benchmark results show that this helps startup/first-request latency.

Do not warm models for every image.

Keep model instances alive during active attendance sessions.

Release resources when appropriate under memory pressure.

---

# 26. Memory Management

Large camera images are expensive.

Avoid retaining all of these simultaneously:

```text
full-resolution source
resized source
all face crops
all intermediate tensors
annotated copy
encoded output
```

Process and release temporary buffers promptly.

For two photos, prefer:

```text
process image 1
release temporary data
process image 2
merge results
```

Do not keep unnecessary images in memory.

---

# 27. Storage Strategy

Store locally:

```text
classes
subjects
students
face profiles
embeddings
attendance
settings
```

Start with `float32` normalized embeddings unless there is a measured reason to change.

Later benchmark `float16` if supported, but do not quantize embeddings blindly.

A recognition model can be quantized while embeddings remain float32.

---

# 28. Security and Privacy

Face embeddings are biometric-derived data and must be treated as sensitive.

Implement:

```text
local-only by default
minimal data retention
explicit export
explicit deletion
no automatic cloud upload
```

Recommended protections:

- encrypt sensitive local data where practical
- protect backups
- never log raw embeddings
- never log raw face images
- do not include biometric data in crash reports
- provide a clear delete-face-data action
- document on-device processing

Do not silently upload anything.

---

# 29. Permissions

Request only what is required.

Likely:

```text
Camera
Photo/media access where required by the Android version and chosen picker
```

Do not request unrelated permissions such as:

```text
Location
Contacts
Microphone
SMS
Phone
```

unless a real future feature requires them.

---

# 30. Offline Verification

Final QA must include a full airplane-mode run.

Test sequence:

```text
Enable airplane mode
 ↓
Force close app
 ↓
Launch app
 ↓
Open class
 ↓
Enroll student
 ↓
Capture attendance
 ↓
Review results
 ↓
Save attendance
 ↓
Open history
 ↓
Export data
```

No network failure should affect this flow.

Use network inspection during QA to verify that the attendance path makes no requests.

---

# 31. Optional Future Sync

Do not implement before the offline system is stable.

Design the data layer so it can later support:

```text
SQLite
  ↓
Sync queue
  ↓
Connectivity available?
  ↓
FastAPI
  ↓
PostgreSQL
```

Use sync states such as:

```text
LOCAL_ONLY
PENDING_SYNC
SYNCED
SYNC_ERROR
```

Never make synchronization a prerequisite for attendance.

---

# 32. Export

Support local export:

```text
CSV
JSON
```

CSV example:

```text
date,class,subject,roll_number,student,status,confidence
```

Use Android's share mechanism. Export must not require a server.

---

# 33. Backup and Restore

Implement after core attendance is stable.

Backup includes:

```text
classes
subjects
students
embeddings
attendance
settings
```

Prefer not to include raw enrollment photos.

Backups must contain:

```text
backup_version
app_version
model_id
model_version
created_at
data
```

Refuse or migrate incompatible embedding versions safely.

---

# 34. Model Versioning

Every embedding must record:

```text
model_id
model_version
embedding_dimension
preprocessing_version
```

Example:

```text
mobileface_v1
512
preprocess_v1
```

When the recognition model changes:

```text
old embeddings
     ↓
incompatible
```

Require re-enrollment or a verified migration.

Never mix versions silently.

---

# 35. Database Versioning

Store:

```text
app_version
database_schema_version
model_version
```

Implement migrations.

Normal app updates must not delete existing attendance or enrollment data.

---

# 36. Benchmark Dataset

Build a real project evaluation set.

Minimum starting point:

```text
20+ registered people
3–5 enrollment images/person
10+ unseen images/person
10+ group images
5+ unknown people
```

Include:

```text
front faces
small faces
side faces
different lighting
different camera distances
glasses
partial occlusion
varied backgrounds
multiple phone cameras
```

Include classroom rows:

```text
front row
middle row
back row
```

---

# 37. Ground-Truth Evaluation

Every test must have a known expected answer.

Measure:

```text
face detection recall
face detection precision
recognition accuracy
false accept rate
false reject rate
unknown detection
review rate
```

Also measure:

```text
per-face latency
per-image latency
full attendance-session latency
```

Do not use one accuracy number as the only metric.

---

# 38. Threshold Calibration

Generate distributions for:

```text
genuine pairs
impostor pairs
```

Sweep candidate thresholds and record:

```text
false accepts
false rejects
review volume
```

Use configurable:

```text
MATCH_THRESHOLD
REVIEW_THRESHOLD
MIN_MARGIN
```

The chosen values must come from mobile-model measurements.

---

# 39. False Positive / Unknown Test

Mandatory test:

```text
No gallery students in the image
```

Expected:

```text
UNKNOWN / REVIEW
```

Never:

```text
unknown person → automatic Present
```

---

# 40. Duplicate Handling

A student appearing in two photos must produce exactly one attendance record.

Deduplicate by:

```text
student_id
```

Keep the strongest valid recognition evidence for audit/review.

---

# 41. Orientation Handling

Test:

```text
portrait
landscape
rotated images
EXIF orientation
```

Normalize orientation before detection.

Phone camera orientation errors must not break attendance.

---

# 42. Image Quality Handling

Start with deterministic checks for:

```text
face size
blur
brightness
pose
occlusion
```

Use the result to:

```text
accept
review
reject/retake
```

Avoid a large complex quality model in v1 unless measurements justify it.

---

# 43. Error Handling

Safely handle:

```text
model load failure
model inference failure
database failure
image decode failure
camera failure
no faces
too many faces
tiny faces
invalid image
out of memory
```

Example rule:

```text
Model unavailable
    ↓
show explicit error
    ↓
never fabricate attendance
```

If ML processing fails, do not automatically mark everyone absent.

---

# 44. Attendance Safety Rules

Never automatically mark:

```text
REVIEW → PRESENT
UNKNOWN → PRESENT
ML error → ABSENT
missing image → ABSENT
```

Safe logic:

```text
MATCH → Present candidate
REVIEW → teacher decides
UNKNOWN → not Present
ML error → retry/review
```

For absence, compute only after confirmed present decisions and completion of the review flow:

```text
Absent = Registered - Confirmed Present
```

---

# 45. Attendance State Machine

Use an explicit state model:

```text
IDLE
CAPTURING
PREPARING
DETECTING
RECOGNIZING
MERGING
REVIEW_REQUIRED
CONFIRMING
SAVING
COMPLETED
ERROR
```

Do not control the workflow with scattered booleans.

---

# 46. Suggested Services

Implement services with clear responsibilities:

```text
LocalDatabaseService
StudentRepository
ClassRepository
SubjectRepository
AttendanceRepository

CameraService
ImagePickerService
ImagePreprocessor

FaceDetectorService
FaceAlignmentService
FaceRecognitionService
EmbeddingStore
GalleryIndexService
FaceMatcher

AttendanceEngine
AttendanceReviewService

ExportService
BackupService
SettingsService

ModelManager
PerformanceMonitor
```

---

# 47. ML Service API

Keep UI independent of the selected runtime.

Conceptual interface:

```text
initialize()
detectFaces(image)
extractEmbedding(faceCrop)
recognize(embedding, gallery)
recognizeGroup(image)
dispose()
```

Group result fields:

```text
faceIndex
bbox
studentId
studentName
score
margin
status
quality
```

---

# 48. Model Manager

Create one model manager responsible for:

```text
load detector
load recognizer
warm up
report model version
report embedding dimension
release resources
```

Screens must not initialize models themselves.

---

# 49. Gallery Index Service

Responsibilities:

```text
load embeddings
normalize embeddings
create contiguous matrix
store student mapping
create offsets
cache index
invalidate after enrollment changes
```

When enrollment changes:

```text
DB update
 ↓
invalidate index
 ↓
rebuild once
```

Do not rebuild the gallery after every face comparison.

---

# 50. Attendance Engine

Input:

```text
registered students
recognition results
teacher review decisions
```

Output:

```text
present
absent
review
unknown
```

Rules:

```text
high-confidence MATCH
    → Present candidate

teacher-confirmed REVIEW
    → Present or Absent

UNKNOWN
    → not Present

unresolved REVIEW
    → keep Review
```

---

# 51. Offline Data Integrity

Use database transactions for:

```text
student enrollment
attendance session creation
attendance record save
backup restore
```

Commit all related changes or none.

Do not leave half-created sessions.

---

# 52. Crash Recovery

If the app closes during processing:

```text
session status = PROCESSING
```

On next launch:

```text
find stale session
 ↓
mark recoverable/failed
 ↓
allow retry
```

Never infer attendance from an incomplete session.

---

# 53. Battery and Thermal Testing

Run repeated attendance workloads such as:

```text
10 sessions
20 sessions
50 sessions
```

Measure:

```text
battery change
temperature
latency degradation
throttling
```

Test sustained behavior, not just the first inference.

---

# 54. Device Compatibility Testing

At minimum test:

```text
low-end Android
mid-range Android
high-end Android
```

Where possible also test different SoCs/vendors.

Test:

```text
camera
ML initialization
face detection
recognition
SQLite
backup/restore
export
release build
```

---

# 55. Hardware Acceleration

After CPU execution is stable, benchmark available acceleration options:

```text
CPU
GPU delegate
NNAPI/NPU where supported
```

Use graceful fallback:

```text
preferred delegate
      ↓
unsupported/error
      ↓
CPU
```

Do not make a hardware-specific delegate mandatory.

Google documents device/vendor-specific LiteRT delegate options, including NPU-related acceleration, but support varies by hardware:
https://ai.google.dev/edge/litert/android/npu

---

# 56. Release Size Optimization

Build release artifacts and measure actual sizes.

Use:

```bash
flutter build appbundle --release
```

For APK size testing:

```bash
flutter build apk --release --split-per-abi
```

Flutter's Android release guidance notes that split-per-ABI APKs avoid bundling native binaries for architectures a device does not use, reducing direct APK size:
https://docs.flutter.dev/deployment/android

Before release remove:

```text
unused assets
unused fonts
sample images
training data
debug assets
unused model files
unnecessary native ABIs
```

Enable Android code shrinking/optimization where safe and retest ML functionality afterward.

---

# 57. Model Packaging Strategy

Bundle the required models with the application instead of downloading them on first launch.

Example:

```text
assets/models/face_detector.tflite
assets/models/face_recognizer.tflite
```

Use the actual format selected during model evaluation.

The installed app must have everything needed for offline attendance immediately after installation.

---

# 58. Model License Audit

Before bundling any model, record:

```text
model name
source
license
commercial-use status
redistribution rights
modification rights
training-data restrictions
```

Do not ship an attractive benchmark result with an incompatible license.

The InsightFace model-zoo documentation includes a non-commercial research-use notice for its listed pretrained model packs. Resolve licensing independently before using those packs in a product:
https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md

---

# 59. Development Milestones

## Milestone 1 — Offline ML proof of concept

Deliver:

```text
Flutter screen
+
local image
+
offline detector
+
offline recognizer
+
embedding generation
+
local matcher
```

Acceptance:

```text
airplane mode works
```

## Milestone 2 — Enrollment

Deliver:

```text
student creation
face enrollment
quality checks
embedding storage
```

Acceptance:

```text
3–5 valid samples/student
```

## Milestone 3 — Group recognition

Deliver:

```text
class gallery
group photo
multi-face detection
multi-face recognition
MATCH/REVIEW/UNKNOWN
```

Acceptance:

```text
20+ faces in a group photo
```

## Milestone 4 — Attendance engine

Deliver:

```text
Present
Absent
Review
Unknown
```

Acceptance:

```text
two-photo deduplication
```

## Milestone 5 — Review UI

Deliver:

```text
uncertain faces
manual confirmation
manual corrections
```

## Milestone 6 — History/export

Deliver:

```text
attendance history
CSV
JSON
local sharing
```

## Milestone 7 — Backup/restore

Deliver:

```text
versioned local backup
restore validation
```

## Milestone 8 — Optimization

Deliver:

```text
benchmarking
image downscale
batch processing
background inference
memory optimization
release-size optimization
```

## Milestone 9 — Release

Deliver:

```text
signed release
AAB/APK
size report
accuracy report
offline test report
device compatibility report
```

---

# 60. Acceptance Criteria

## Offline

- [ ] Works with airplane mode enabled.
- [ ] No server required for attendance.
- [ ] ML inference runs on device.
- [ ] Local database works offline.
- [ ] Export works offline.

## Recognition

- [ ] Multi-face detection works.
- [ ] Unknown people are handled safely.
- [ ] REVIEW state works.
- [ ] Enrollment and attendance use the same mobile model/version.
- [ ] Embeddings from incompatible models are rejected.

## Attendance

- [ ] Present calculation works.
- [ ] Absent calculation works.
- [ ] Review workflow works.
- [ ] Two-photo deduplication works.
- [ ] Manual corrections persist.

## Performance

- [ ] Models load once.
- [ ] Warmup is one-time only where useful.
- [ ] Large images are downscaled.
- [ ] UI stays responsive.
- [ ] Benchmark results are recorded.
- [ ] Device-tier testing is completed.

## Storage

- [ ] SQLite persistence works.
- [ ] Database migrations work.
- [ ] Backup works.
- [ ] Restore works.
- [ ] CSV/JSON export works.

## Privacy/security

- [ ] No unnecessary permissions.
- [ ] No raw biometric logs.
- [ ] Face-data deletion exists.
- [ ] Backup handling is documented.
- [ ] No silent network upload.

## Package size

- [ ] Release artifact measured.
- [ ] Under 100 MB target.
- [ ] Unnecessary ABIs removed from direct APK tests.
- [ ] Training/debug/sample assets excluded.
- [ ] ML runtime/model files audited.

---

# 61. Recommended Coding-Agent Execution Order

The coding agent must execute the work in this order:

```text
PHASE 0
Inspect existing repository and requirements
        ↓
PHASE 1
Select/benchmark mobile ML runtime + models
        ↓
PHASE 2
Build minimal offline ML prototype
        ↓
PHASE 3
Implement student enrollment + local embeddings
        ↓
PHASE 4
Implement multi-face/group recognition
        ↓
PHASE 5
Implement attendance engine
        ↓
PHASE 6
Implement teacher review UI
        ↓
PHASE 7
Implement SQLite/history/export
        ↓
PHASE 8
Optimize latency, memory, battery
        ↓
PHASE 9
Security/privacy review
        ↓
PHASE 10
Release-size optimization
        ↓
PHASE 11
Real-device/offline testing
```

Do not begin extensive UI polish before offline ML inference and enrollment are working.

---

# 62. Master Coding-Agent Prompt

Use the following as the primary instruction for the coding agent:

```text
You are implementing the mobile application for the AI Classroom Attendance project.

OBJECTIVE
Build an Android-first Flutter application that performs student enrollment, classroom multi-face detection, fully offline face recognition, attendance calculation, manual review, history, export, backup/restore, and local persistence.

PRIMARY CONSTRAINTS
1. Attendance must work with airplane mode enabled.
2. ML inference must run on-device.
3. No server is required for the attendance path.
4. Final release target is <100 MB.
5. Do not bundle Python.
6. Do not bundle the desktop InsightFace Python stack.
7. Do not blindly use buffalo_l/buffalo_s model packs in the mobile application.
8. Select a lightweight mobile detector and recognition model through benchmarking.
9. Use the exact same mobile recognition model/preprocessing for enrollment and attendance.
10. Never mix incompatible embedding spaces.
11. Unknown and uncertain faces must never automatically become Present.
12. Keep the existing backend optional for future sync only.

IMPLEMENTATION PROCESS
1. Inspect the repository before editing files.
2. Identify existing API/data contracts that should remain compatible with future sync.
3. Build a small offline ML benchmark before implementing the entire UI.
4. Compare candidate detectors and recognizers using real project images.
5. Measure model size, startup, warm inference, RAM, detection recall, recognition accuracy, false accepts/rejects, and classroom behavior.
6. Select the smallest model combination that meets the project accuracy/performance requirements.
7. Record the chosen model ID/version/license in the repository.
8. Implement model loading behind ModelManager.
9. Load each model once and warm it once if useful.
10. Implement centralized preprocessing used by both enrollment and attendance.
11. Implement local SQLite persistence.
12. Implement student enrollment with 3–5 quality-controlled samples.
13. Generate normalized embeddings locally.
14. Store model/version metadata with embeddings.
15. Implement GalleryIndex with pre-normalized embeddings and efficient vectorized matching.
16. Implement MATCH/REVIEW/UNKNOWN logic using calibrated mobile-model thresholds.
17. Implement one- and two-photo attendance with student-ID deduplication.
18. Implement teacher review before finalizing ambiguous cases.
19. Calculate Absent only from the registered roster minus confirmed Present students.
20. Implement local history, CSV/JSON export, and versioned backup/restore.
21. Test entirely in airplane mode.
22. Optimize image size, inference, memory, and UI responsiveness.
23. Build release artifacts and verify the <100 MB target.
24. Test on low/mid/high Android devices.
25. Do not declare completion until the acceptance criteria are tested.

ARCHITECTURE
Flutter
→ camera/image picker
→ image preprocessing
→ lightweight face detector
→ face quality check
→ face alignment
→ lightweight recognition encoder
→ normalized embedding
→ local GalleryIndex
→ similarity/margin decision
→ MATCH / REVIEW / UNKNOWN
→ attendance engine
→ SQLite

DATA MODEL
Use local SQLite for classes, subjects, students, face profiles, embeddings, attendance sessions, attendance records, and optional recognition events.

ENROLLMENT
- Register a student.
- Capture 3–5 good samples.
- Require a usable face in each enrollment sample.
- Reject/retake on poor quality.
- Generate embeddings on device.
- Normalize and store them with model metadata.

GROUP RECOGNITION
- Accept one or two classroom images.
- Correct EXIF orientation.
- Downscale oversized images.
- Detect all faces.
- Reject/flag unusably tiny faces.
- Generate embeddings.
- Match locally.
- Calculate score and margin.
- Produce MATCH/REVIEW/UNKNOWN.
- Deduplicate recognized student IDs across photos.

ATTENDANCE
MATCH → Present candidate
REVIEW → teacher decision
UNKNOWN → not Present
ML error → retry/review
Absent = Registered - Confirmed Present

PERFORMANCE
- Keep model instances alive while active.
- Use background execution so Flutter UI remains responsive.
- Avoid unnecessary copies.
- Downscale large images.
- Vectorize gallery matching.
- Benchmark CPU first.
- Evaluate hardware delegates after CPU correctness.
- Add no heavy optimization library until measurements justify it.

PACKAGE SIZE
- Bundle only required ML models.
- Remove unused assets.
- Remove training/test datasets.
- Use appropriate ABI strategy.
- Use release/code shrinking where safe.
- Measure final artifact, do not estimate it.

PRIVACY
- No cloud recognition.
- No required network calls.
- Avoid raw face-image retention.
- Do not log embeddings.
- Provide deletion/export controls.
- Document local processing.

TESTING
- Unit-test similarity, thresholds, margins, deduplication, and attendance logic.
- Integration-test enrollment, one-photo attendance, two-photo attendance, review, history, backup/restore, and airplane-mode behavior.
- Benchmark real images containing known and unknown people.
- Include small/side/occluded faces.

FINAL REPORT
Provide:
1. Selected model/runtime and reason.
2. Model sizes.
3. Cold/warm performance.
4. Detection recall.
5. Recognition metrics.
6. Unknown-person results.
7. Memory usage.
8. Battery/thermal observations.
9. Final release package size.
10. Device compatibility.
11. Known limitations.
12. Build/install instructions.

Do not claim success because the project compiles. Verify the real acceptance criteria.
```

---

# 63. Final Product Vision

The finished app should feel like this:

```text
Teacher opens app
        ↓
Select class + subject
        ↓
Start Attendance
        ↓
Capture classroom photo
        ↓
Local processing
        ↓
30 faces detected
        ↓
28 high-confidence matches
2 review items
        ↓
Teacher resolves review
        ↓
Attendance finalized
        ↓
SQLite save
        ↓
History / Export
```

With:

```text
Internet: OFF
Server: not required
Recognition: on-device
Database: local
Primary face images: not required after processing
Target app size: <100 MB
```

The mobile application should be treated as a purpose-built on-device inference system, not as a thin UI wrapper around the existing desktop/server pipeline.

---

## Official / Primary References

- Flutter Android setup and release: https://docs.flutter.dev/deployment/android
- Flutter Android development setup: https://docs.flutter.dev/platform-integration/android/setup
- MediaPipe Face Detector: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceDetector
- MediaPipe Face Detector options: https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/FaceDetectorOptions
- ONNX Runtime mobile: https://onnxruntime.ai/docs/tutorials/mobile/
- ONNX Runtime Mobile: https://onnxruntime.ai/docs/get-started/with-mobile.html
- LiteRT Android NPU delegates: https://ai.google.dev/edge/litert/android/npu
- InsightFace model zoo and model-pack information: https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
