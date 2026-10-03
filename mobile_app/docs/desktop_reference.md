# Desktop Reference Behavior (Mobile App Design Input)

> Source of truth: `../app.py`, `../registration_app.py`, `../benchmark_results.txt`
> Purpose: capture the behavior the mobile app must replicate conceptually,
> while replacing InsightFace/buffalo_l with a mobile model (Section 5 of plan).

## Pipeline (desktop)

```
upload -> EXIF transpose -> RGB
       -> downscale (MAX_IMAGE_SIZE=1280, INTER_AREA, never upscale)
       -> InsightFace FaceAnalysis (buffalo_l, det 640x640, CPU)
       -> per-face embedding (512-d float32, from model)
       -> GalleryIndex match (vectorized)
       -> MATCH / REVIEW / UNKNOWN
```

## Thresholds (desktop — DO NOT copy to mobile)

| Constant | Value | Where |
|---|---|---|
| MATCH_THRESHOLD | 0.55 | `app.py` |
| MIN_MARGIN | 0.05 | `app.py` |
| REVIEW_THRESHOLD | 0.45 (hardcoded) | `_classify_scores` |
| MAX_IMAGE_SIZE | 1280 | `app.py` |
| DET_SIZE | 640x640 | `app.py` |

Classification: `score >= 0.55 && margin >= 0.05 -> MATCH; score >= 0.45 -> REVIEW; else UNKNOWN`.

## Gallery format (desktop)

`gallery/multi_sample_gallery.pkl`: `dict[student_id -> {name, embeddings: list[list[float]]}]`
- 43 registered students (`gallery/profiles.json`: face_01..face_43)
- Embeddings stored pre-normalized (L2), float32, 512-dim
- `face_gallery.pkl` is the legacy single-sample file (ignore)

## Matching algorithm (desktop, replicate in Dart/Kotlin)

1. Normalize all gallery embeddings once at load.
2. Stack into one (N, 512) float32 matrix + per-student row offsets (`starts`).
3. Queries normalized, then `queries @ matrix.T` (one matmul).
4. Per-student best score via max-reduce over each student's row block.
5. `margin = best - second_best_student` (1.0 if only one student).
6. Classify with thresholds above.

## Enrollment (desktop)

- Group photo -> detect all faces -> skip existing MATCHes.
- New faces get padded crops (30% x-pad, 40% y-pad), normalized embedding saved per profile.
- Atomic gallery write via temp file + rename.

## Known benchmark numbers (desktop, 43 test images)

| Config | Mean/img | Detection | Matching |
|---|---|---|---|
| buffalo_l 640x640 | 6.35 s | 6346 ms | 0.53 ms |
| buffalo_s 640x640 | 1.72 s | 1711 ms | 1.11 ms |
| buffalo_s 320x320 | 0.73 s | 729 ms | 2.13 ms |

Gallery matching is negligible vs detection — same expectation on mobile
(per Section 24: do not prematurely optimize the gallery).

## Contracts to preserve for future sync (Section 61, Phase 0)

- Student identity: string student_id (e.g. "101"), human-readable name.
- Recognition result shape: {student_id, name, score, margin, status}.
- Status vocabulary: MATCH / REVIEW / UNKNOWN (display) and
  PRESENT / ABSENT / REVIEW (attendance records).
- Mobile embeddings are a NEW space: mobile model id + version must be stored
  with every embedding; never mix with buffalo_l vectors.
