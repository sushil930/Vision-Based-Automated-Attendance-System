import 'dart:math' as math;
import 'dart:typed_data';

import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../core/constants/pipeline_config.dart';
import 'matcher/gallery_index.dart';
import 'matcher/match_result.dart';
import 'preprocessing/image_preprocessor.dart';

/// Face bounding box in prepared-image coordinates plus eye landmarks for
/// alignment (Section 47 group result fields).
class DetectedFace {
  const DetectedFace({
    required this.left,
    required this.top,
    required this.right,
    required this.bottom,
    this.eyeLeftDx,
    this.eyeLeftDy,
    this.eyeRightDx,
    this.eyeRightDy,
  });

  final double left, top, right, bottom;
  final double? eyeLeftDx, eyeLeftDy, eyeRightDx, eyeRightDy;

  double get width => right - left;
  double get height => bottom - top;
  double get area => width * height;
}

/// One classified face for a group result (Section 47).
class GroupFaceResult {
  const GroupFaceResult({
    required this.faceIndex,
    required this.bbox,
    required this.status,
    required this.score,
    required this.margin,
    this.studentId,
    this.studentName,
    this.qualityNote,
  });

  final int faceIndex;
  final DetectedFace bbox;
  final RecognitionStatus status;
  final double score;
  final double margin;
  final String? studentId;
  final String? studentName;
  final String? qualityNote;
}

/// Loads the recognizer once, warms up once, reports versions, releases
/// resources (Section 48). Screens never initialize models themselves.
class ModelManager {
  ModelManager({this.recognizerAsset = 'models/face_recognizer.tflite'});

  final String recognizerAsset;

  Interpreter? _recognizer;
  bool _initialized = false;

  bool get isInitialized => _initialized;
  int get embeddingDimension => PipelineConfig.embeddingDimension;
  String get modelId => PipelineConfig.modelId;
  String get modelVersion => PipelineConfig.modelVersion;

  /// Loads the recognizer once and runs one dummy inference so the first real
  /// request is not charged for init (desktop parity, Section 25).
  Future<void> initialize() async {
    if (_initialized) {
      return;
    }
    final interpreter = await Interpreter.fromAsset(
      recognizerAsset,
      options: InterpreterOptions()..threads = 4,
    );
    _recognizer = interpreter;
    _initialized = true;

    // One-time warmup on a zero input.
    final input = Float32List(
      interpreter.getInputTensor(0).shape.fold<int>(1, (a, b) => a * b),
    );
    final out = Float32List(embeddingDimension);
    interpreter.getInputTensor(0).setTo(input);
    interpreter.invoke();
    interpreter.getOutputTensor(0).copyTo(out);
  }

  Interpreter get recognizer {
    final r = _recognizer;
    if (r == null) {
      throw const ModelNotInitializedException();
    }
    return r;
  }

  Future<void> dispose() async {
    _recognizer?.close();
    _recognizer = null;
    _initialized = false;
  }
}

class ModelNotInitializedException implements Exception {
  const ModelNotInitializedException();

  @override
  String toString() => 'ModelNotInitializedException: recognizer not loaded';
}

/// Face detection via MLKit (BlazeFace full-range in accurate mode).
/// Multi-face by design; configurable thresholds (Section 11).
class FaceDetectorService {
  FaceDetectorService._(this._detector);

  static FaceDetectorService? _instance;

  static FaceDetectorService instance() {
    _instance ??= FaceDetectorService._(
      FaceDetector(
        options: FaceDetectorOptions(
          performanceMode: FaceDetectorMode.accurate,
          enableLandmarks: true,
          enableClassification: false,
          enableTracking: false,
          minFaceSize: PipelineConfig.minDetectionConfidence,
        ),
      ),
    );
    return _instance!;
  }

  final FaceDetector _detector;

  /// Detect on a file; MLKit applies EXIF orientation itself (Section 41).
  Future<List<DetectedFace>> detectFile(String path) async {
    final inputImage = InputImage.fromFilePath(path);
    final faces = await _detector.processImage(inputImage);
    return faces.map(_toDetectedFace).toList();
  }

  DetectedFace _toDetectedFace(Face face) {
    double? ldx, ldy, rdx, rdy;
    final lm = face.landmarks;
    final leftEye = lm[FaceLandmarkType.leftEye];
    final rightEye = lm[FaceLandmarkType.rightEye];
    if (leftEye != null) {
      ldx = leftEye.position.x.toDouble();
      ldy = leftEye.position.y.toDouble();
    }
    if (rightEye != null) {
      rdx = rightEye.position.x.toDouble();
      rdy = rightEye.position.y.toDouble();
    }
    return DetectedFace(
      left: face.boundingBox.left.toDouble(),
      top: face.boundingBox.top.toDouble(),
      right: face.boundingBox.right.toDouble(),
      bottom: face.boundingBox.bottom.toDouble(),
      eyeLeftDx: ldx,
      eyeLeftDy: ldy,
      eyeRightDx: rdx,
      eyeRightDy: rdy,
    );
  }

  Future<void> dispose() => _detector.close();
}

/// Face embedding extraction: model input -> interpreter -> L2 norm.
class FaceRecognizerService {
  FaceRecognizerService(this._manager);

  final ModelManager _manager;

  static const int inputSize = 112;

  /// Extract an L2-normalized embedding for one preprocessed face input.
  List<double> extractEmbedding(Float32List modelInput) {
    final interpreter = _manager.recognizer;
    interpreter.getInputTensor(0).setTo(modelInput);
    interpreter.invoke();
    final out = Float32List(_manager.embeddingDimension);
    interpreter.getOutputTensor(0).copyTo(out);

    var norm = 0.0;
    for (final v in out) {
      norm += v * v;
    }
    norm = norm <= 0 ? 1.0 : math.sqrt(norm);
    return [for (final v in out) v / norm];
  }
}

/// Progress states surfaced to the UI (Section 21).
enum RecognitionStage { preparing, detecting, recognizing, merging }

/// Facade over detect -> quality -> align -> embed -> match (Section 47 API).
class FaceRecognitionService {
  FaceRecognitionService({required ModelManager modelManager})
      : _manager = modelManager;

  final ModelManager _manager;
  GalleryIndex? _gallery;
  final ImagePreprocessor _preprocessor = const ImagePreprocessor();

  set gallery(GalleryIndex? index) => _gallery = index;

  /// Full group pipeline for one image path (Section 22).
  Future<List<GroupFaceResult>> recognizeGroupImage(
    String path, {
    required double matchThreshold,
    required double reviewThreshold,
    required double minMargin,
    void Function(RecognitionStage stage)? onStage,
  }) async {
    onStage?.call(RecognitionStage.preparing);
    final prepared = await _preprocessor.prepareFullImage(path);
    try {
      onStage?.call(RecognitionStage.detecting);
      final detector = FaceDetectorService.instance();
      final faces = await detector.detectFile(path);
      if (faces.isEmpty) {
        return const [];
      }

      onStage?.call(RecognitionStage.recognizing);
      final recognizer = FaceRecognizerService(_manager);
      final results = <GroupFaceResult>[];
      final embeddings = <List<double>>[];
      final keptFaces = <DetectedFace>[];

      for (int i = 0; i < faces.length; i++) {
        final face = faces[i];
        // Quality gate: skip hopeless crops, send to review (Section 11).
        if (face.width < PipelineConfig.minFaceSizePx ||
            face.height < PipelineConfig.minFaceSizePx) {
          results.add(GroupFaceResult(
            faceIndex: i,
            bbox: face,
            status: RecognitionStatus.review,
            score: 0,
            margin: 0,
            qualityNote: 'face too small',
          ));
          continue;
        }
        final crop = await _preprocessor.cropFace(
          prepared,
          left: face.left,
          top: face.top,
          right: face.right,
          bottom: face.bottom,
          eyeLeftDx: face.eyeLeftDx,
          eyeLeftDy: face.eyeLeftDy,
          eyeRightDx: face.eyeRightDx,
          eyeRightDy: face.eyeRightDy,
        );
        final input = await _preprocessor.toModelInput(
            crop, FaceRecognizerService.inputSize);
        crop.dispose();
        embeddings.add(recognizer.extractEmbedding(input));
        keptFaces.add(face);
      }

      onStage?.call(RecognitionStage.merging);
      final matches = _gallery?.matchBatch(
            embeddings,
            matchThreshold: matchThreshold,
            reviewThreshold: reviewThreshold,
            minMargin: minMargin,
          ) ??
          List.generate(
              embeddings.length, (_) => _unknownMatch);

      int keptIdx = 0;
      for (int i = 0; i < faces.length; i++) {
        final alreadyMarked = results.any((r) => r.faceIndex == i);
        if (alreadyMarked) {
          continue;
        }
        final m = matches[keptIdx++];
        results.add(GroupFaceResult(
          faceIndex: i,
          bbox: keptFaces[keptIdx - 1],
          status: m.status,
          score: m.score,
          margin: m.margin,
          studentId: m.status == RecognitionStatus.unknown ? null : m.studentId,
          studentName: m.status == RecognitionStatus.unknown ? null : m.name,
        ));
      }
      results.sort((a, b) => a.faceIndex.compareTo(b.faceIndex));
      return results;
    } finally {
      _preprocessor.dispose(prepared);
    }
  }

  static const _unknownMatch = MatchResult(
    studentId: '-',
    name: 'Unknown',
    score: -1.0,
    margin: 1.0,
    status: RecognitionStatus.unknown,
  );
}
