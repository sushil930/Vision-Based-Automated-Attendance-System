# Model Selection Decision Log

> Plan sections: 4 (ML runtime), 6 (benchmark phase), 58 (license audit).
> Every field below must be filled before Phase 2 model bundling.

## Decision record (fill during Phase 1)

| Field | Value |
|---|---|
| Selected runtime | **LiteRT / tflite_flutter 0.11.0** (provisional — pending on-device benchmark vs ONNX Runtime Mobile) |
| Runtime version | tflite_flutter 0.11.0 (Flutter 3.47.6 / Dart 3.13.5) |
| Selected detector | **MLKit Face Detection** (BlazeFace full-range, accurate mode, on-device) |
| Detector file | bundled inside native MLKit lib (no asset needed) |
| Detector size | ~0 MB app-side (in-play-services-free on-device lib) |
| Selected recognizer | **MobileFaceNet** (TFLite conversion of the 9925_9680 checkpoint) |
| Recognizer file | `assets/models/face_recognizer.tflite` |
| Recognizer size | 5,111 KB (5.0 MB) |
| model_id | `mobileface_v1` |
| model_version | `1` |
| embedding_dimension | 512 |
| preprocessing_version | `preprocess_v1` |
| input size | 112x112 NHWC float32 |
| normalization | (px - 127.5) / 127.5, RGB channel order |
| alignment | eye-line rotation via MLKit eye landmarks, desktop-parity padding (30% x / 40% y) |

> STATUS: provisional. Per plan Section 6, the on-device benchmark
> (startup, warm inference, RAM, recall on small/side/occluded faces, and
> genuine/impostor distributions for threshold calibration, Section 38) must
> run on real hardware before thresholds ship. Placeholder thresholds mirror
> desktop values ONLY to keep the pipeline runnable; they are not final.

## Benchmark results (fill during Phase 1)

### Detectors

| Candidate | Size | Startup | Warm infer | RAM | Recall small/side/occluded | Notes |
|---|---|---|---|---|---|---|
| MediaPipe Face Detector | | | | | | |
| SCRFD-mobile | | | | | | |
| BlazeFace-class | | | | | | |

### Recognizers

| Candidate | Size | Cold infer | Warm infer | Genuine mean | Impostor mean | FAR@thr | FRR@thr |
|---|---|---|---|---|---|---|---|
| MobileFaceNet | | | | | | | |
| (other) | | | | | | | |

### Runtimes

| Runtime | Native dep size | Cold init | Warm infer | RAM | Delegates | Integration |
|---|---|---|---|---|---|---|
| LiteRT/TFLite | | | | | | |
| ONNX Runtime Mobile | | | | | | |

## License audit (Section 58 — required before bundling)

| Model | Source | License | Commercial use | Redistribution | Modification | Training-data restrictions |
|---|---|---|---|---|---|---|
| _TBD_ | | | | | | |

> InsightFace model-zoo packs (buffalo_l/s) are non-commercial research-only.
> They are NOT candidates. Do not bundle them.

## Rejected candidates + reasons

| Candidate | Reason rejected |
|---|---|
| buffalo_l / buffalo_s | ~326 MB / ~159 MB packs; non-commercial license; violates <100 MB |
