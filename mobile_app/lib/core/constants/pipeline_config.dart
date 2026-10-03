/// Central pipeline configuration (single source of truth, mirrors desktop app.py).
///
/// Section 12: preprocessing is centralized here and reused by BOTH enrollment
/// and attendance. Section 15: thresholds must be recalibrated for the mobile
/// model — desktop values below are temporary placeholders, never final.
class PipelineConfig {
  PipelineConfig._();

  /// Longest-edge cap before detection. Plan Section 13 start value.
  /// Benchmark 960/1280/1600/1920 and keep the lowest that preserves recall.
  static const int maxImageSize = 1280;

  /// Never upscale small images (plan Section 13).
  static const bool allowUpscale = false;

  /// Model identity recorded with every stored embedding (Section 34).
  /// TODO(Phase1): replace with values measured in the model benchmark.
  static const String modelId = 'mobileface_v1';
  static const String modelVersion = '1';
  static const int embeddingDimension = 512;
  static const String preprocessingVersion = 'preprocess_v1';

  /// Recognition decision thresholds (Section 15).
  ///
  /// PLACEHOLDER VALUES copied from the desktop buffalo_l pipeline only to
  /// make the pipeline runnable end-to-end. Must be recalibrated from the
  /// mobile model's own genuine/impostor score distributions (Section 38).
  static const double matchThreshold = 0.55;
  static const double reviewThreshold = 0.45;
  static const double minMargin = 0.05;

  /// Detection tuning (Section 11). Final values must come from benchmarks.
  static const double minDetectionConfidence = 0.5;
  static const int minFaceSizePx = 40;
  static const int maxFaces = 100;

  /// Enrollment sample strategy (Section 9): store 3–5 samples, no averaging.
  static const int minEnrollmentSamples = 3;
  static const int maxEnrollmentSamples = 5;
}
