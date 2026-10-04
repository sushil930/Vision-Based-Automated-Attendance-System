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
  // Path must include the 'assets/' prefix: tflite_flutter's fromAsset
  // resolves it via rootBundle.load(assetFileName) directly.
  ModelManager({this.recognizerAsset = 'assets/models/face_recognizer.tflite'});

  final String recognizerAsset;

  Interpreter? _recognizer;
  bool _initialized = false;
  List<int> _inputShape = const [1, 112, 112, 3];
  List<int> _outputShape = const [1, PipelineConfig.embeddingDimension];
  int _inputBatch = 1;

  bool get isInitialized => _initialized;

  /// Embedding dimension read from the loaded model at runtime (falls back
  /// to PipelineConfig before initialization).
  int get embeddingDimension =>
      _outputShape.isEmpty ? PipelineConfig.embeddingDimension : _outputShape.last;

  /// Batch axis of the (possibly resized) model input.
  int get inputBatch => _inputBatch;

  /// Total float elements of one batched model input.
  int get inputFlatLength => _inputShape.fold<int>(1, (a, b) => a * b);

  /// Total float elements of one batched model output.
  int get outputFlatLength => _outputShape.fold<int>(1, (a, b) => a * b);

  /// Loads the recognizer once and runs one dummy inference so the first real
  /// request is not charged for init (desktop parity, Section 25).
  Future<void> initialize() async {
    if (_initialized) {
      return;
    }
    // Interpreter._create allocates tensors with the model's native shapes.
    final interpreter = await Interpreter.fromAsset(
      recognizerAsset,
      options: InterpreterOptions()..threads = 4,
    );

    _inputShape = interpreter.getInputTensor(0).shape;
    _outputShape = interpreter.getOutputTensor(0).shape;
    // The bundled MobileFaceNet TOCO export hardcodes batch=2 through the
    // whole graph, so resizing the axis breaks AllocateTensors. Keep the
    // native batch and feed duplicated rows instead (see extractEmbedding).
    _inputBatch = _inputShape[0];
    _recognizer = interpreter;
    _initialized = true;

    // One-time warmup on a zero input.
    final input = Float32List(inputFlatLength);
    interpreter.getInputTensor(0).setTo(input);
    interpreter.invoke();
    interpreter.getOutputTensor(0).copyTo(shapedOutput());
  }

  /// Builds a nested output object matching the output tensor's shape —
  /// copyTo requires the destination's shape to equal the tensor's shape
  /// (a flat list of the same length is rejected).
  Object shapedOutput() {
    Object build(List<int> dims) {
      if (dims.length <= 1) {
        return List<double>.filled(dims.isEmpty ? 0 : dims[0], 0);
      }
      return List.generate(dims[0], (_) => build(dims.sublist(1)));
    }

    return build(_outputShape);
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
  /// `modelInput` is a single face ([1, size, size, 3] flat). If the loaded
  /// model kept a fixed batch > 1, the crop is duplicated across the batch
  /// axis and the first output row is used.
  List<double> extractEmbedding(Float32List modelInput) {
    final interpreter = _manager.recognizer;
    final batch = _manager.inputBatch;
    final perItem = modelInput.length;
    final fullInput = batch == 1
        ? modelInput
        : (Float32List(perItem * batch)
          ..setRange(0, perItem, modelInput));
    if (batch > 1) {
      for (int b = 1; b < batch; b++) {
        fullInput.setRange(b * perItem, (b + 1) * perItem, modelInput);
      }
    }
    interpreter.getInputTensor(0).setTo(fullInput);
    interpreter.invoke();
    // copyTo returns the converted object shaped like the output tensor:
    // [batch][dim] for the batch-2 export, [dim] for a batch-1 graph.
    final outObj = interpreter.getOutputTensor(0).copyTo(_manager.shapedOutput());
    final dim = _manager.embeddingDimension;
    final List<double> embedding;
    if (outObj is List && outObj.isNotEmpty && outObj.first is List) {
      // Duplicated batch rows share the same input -> identical outputs.
      embedding = (outObj.first as List).cast<double>();
    } else {
      embedding = (outObj as List).cast<double>();
    }
    if (embedding.length != dim) {
      throw StateError(
          'Recognizer output dim ${embedding.length} != expected $dim');
    }

    var norm = 0.0;
    for (final v in embedding) {
      norm += v * v;
    }
    norm = norm <= 0 ? 1.0 : math.sqrt(norm);
    return [for (final v in embedding) v / norm];
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
