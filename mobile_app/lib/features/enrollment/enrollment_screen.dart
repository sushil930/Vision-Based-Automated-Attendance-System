import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import '../../app/args.dart';
import '../../app/services.dart';
import '../../core/constants/pipeline_config.dart';
import '../../core/errors/ml_errors.dart';
import '../../core/permissions/permission_service.dart';
import '../../ml/face_recognition_service.dart';
import '../../ml/preprocessing/image_preprocessor.dart';

/// Enrollment (Section 9): capture 3–5 quality-controlled samples,
/// generate embeddings on device, store with model metadata.
class EnrollmentScreen extends StatefulWidget {
  const EnrollmentScreen({super.key, required this.args});

  final EnrollmentArgs args;

  @override
  State<EnrollmentScreen> createState() => _EnrollmentScreenState();
}

enum _EnrollStage { capturing, checking, storing, done, error }

class _EnrollmentScreenState extends State<EnrollmentScreen> {
  CameraController? _controller;
  int _captured = 0;
  _EnrollStage _stage = _EnrollStage.capturing;
  String _feedback = '';
  String? _error;
  final List<List<double>> _samples = [];

  @override
  void initState() {
    super.initState();
    _startCamera();
  }

  Future<void> _startCamera() async {
    final permission = await const PermissionService().ensureCamera();
    if (!mounted) return;
    if (!permission.isGranted) {
      setState(() {
        _error = permission == CameraPermissionResult.permanentlyDenied
            ? 'Camera permission is permanently denied. Enable it in system settings.'
            : 'Camera permission was denied. Enrollment needs the camera.';
        _stage = _EnrollStage.error;
      });
      return;
    }
    try {
      final cameras = await availableCameras();
      final front = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        front,
        ResolutionPreset.high,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() => _controller = controller);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = CameraFailure(e.toString()).toString();
        _stage = _EnrollStage.error;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _capture() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    setState(() {
      _stage = _EnrollStage.checking;
      _feedback = 'Checking sample...';
    });

    try {
      final file = await controller.takePicture();
      final services = AppServices.instance;
      final prepared = await const ImagePreprocessor().prepareFullImage(file.path);
      try {
        final faces = await FaceDetectorService.instance().detectFile(file.path);

        // Quality gate (Section 9): exactly one usable face required.
        final usable = faces
            .where((f) =>
                f.width >= PipelineConfig.minFaceSizePx &&
                f.height >= PipelineConfig.minFaceSizePx)
            .toList();
        if (faces.isEmpty) {
          _reject('No face found. Improve lighting and retake.');
          return;
        }
        if (usable.isEmpty) {
          _reject('Face too small. Move closer and retake.');
          return;
        }
        if (usable.length > 1) {
          _reject('Multiple faces found. Capture one person only.');
          return;
        }

        final face = usable.single;
        final crop = await const ImagePreprocessor().cropFace(
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
        final input = await const ImagePreprocessor()
            .toModelInput(crop, FaceRecognizerService.inputSize);
        crop.dispose();

        setState(() => _stage = _EnrollStage.storing);
        final recognizer =
            FaceRecognizerService(AppServices.instance.modelManager);
        final embedding = recognizer.extractEmbedding(input);
        _samples.add(embedding);
        _captured++;
        if (_captured >= PipelineConfig.minEnrollmentSamples) {
          await services.embeddings.saveProfileWithEmbeddings(
            studentId: widget.args.studentId,
            samples: _samples,
          );
          await services.refreshGallery();
          if (!mounted) return;
          setState(() {
            _stage = _EnrollStage.done;
            _feedback =
                'Enrolled ${_samples.length} samples for ${widget.args.studentName}';
          });
        } else {
          if (!mounted) return;
          setState(() {
            _stage = _EnrollStage.capturing;
            _feedback =
                'Sample $_captured/${PipelineConfig.minEnrollmentSamples} accepted. Turn slightly and continue.';
          });
        }
      } finally {
        const ImagePreprocessor().dispose(prepared);
      }
    } on MlError catch (e) {
      _reject(e.toString());
    } catch (e) {
      _reject('Capture failed: $e');
    }
  }

  void _reject(String message) {
    if (!mounted) return;
    setState(() {
      _stage = _EnrollStage.capturing;
      _feedback = message;
    });
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: Text('Enroll ${widget.args.studentName}')),
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, size: 48, color: Colors.red),
              const SizedBox(height: 12),
              const Text('Camera unavailable'),
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(fontSize: 12)),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () {
                  setState(() => _error = null);
                  _startCamera();
                },
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }
    return Scaffold(
      appBar: AppBar(title: Text('Enroll ${widget.args.studentName}')),
      body: Column(
        children: [
          Expanded(
            child: controller != null && controller.value.isInitialized
                ? CameraPreview(controller)
                : const Center(child: CircularProgressIndicator()),
          ),
          if (_feedback.isNotEmpty)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Text(
                _feedback,
                style: TextStyle(
                  color: _feedback.startsWith('Sample') ||
                          _stage == _EnrollStage.done
                      ? Colors.green
                      : Colors.orange,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(24),
            child: _stage == _EnrollStage.done
                ? FilledButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('Finish'),
                  )
                : _stage == _EnrollStage.checking || _stage == _EnrollStage.storing
                    ? const CircularProgressIndicator()
                    : FilledButton.icon(
                        icon: const Icon(Icons.camera),
                        label: Text(
                            'Capture sample (${_captured + 1}/${PipelineConfig.minEnrollmentSamples})'),
                        onPressed: _capture,
                      ),
          ),
        ],
      ),
    );
  }
}
