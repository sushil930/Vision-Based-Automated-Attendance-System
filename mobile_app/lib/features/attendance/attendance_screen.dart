import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../../app/args.dart';
import '../../app/services.dart';
import '../../core/constants/pipeline_config.dart';
import '../../core/permissions/permission_service.dart';
import '../../domain/entities/entities.dart';
import '../../ml/face_recognition_service.dart';
import '../../ml/matcher/match_result.dart';
import '../review/review_screen.dart';

/// Explicit attendance state machine (Section 45). No scattered booleans.
enum AttendanceFlowState {
  idle,
  capturing,
  preparing,
  detecting,
  recognizing,
  merging,
  reviewRequired,
  confirming,
  saving,
  completed,
  error,
}

/// Attendance flow (Sections 17, 20): capture 1–2 photos, process offline,
/// hand uncertain cases to review, then save. No recognition on preview
/// frames (Section 20).
class AttendanceScreen extends StatefulWidget {
  const AttendanceScreen({super.key, required this.args});

  final AttendanceArgs args;

  @override
  State<AttendanceScreen> createState() => _AttendanceScreenState();
}

class _AttendanceScreenState extends State<AttendanceScreen> {
  CameraController? _controller;
  AttendanceFlowState _state = AttendanceFlowState.idle;
  final List<String> _photoPaths = [];
  final List<List<GroupFaceResult>> _photoResults = [];
  List<Student> _roster = const [];
  String _stageText = '';
  String? _error;
  bool get _twoPhotosDone => _photoPaths.length >= 2;

  @override
  void initState() {
    super.initState();
    _loadRoster();
    _startCamera();
  }

  Future<void> _loadRoster() async {
    final roster =
        await AppServices.instance.students.listForClass(widget.args.classId);
    if (!mounted) return;
    setState(() => _roster = roster);
  }

  Future<void> _startCamera() async {
    final permission = await const PermissionService().ensureCamera();
    if (!mounted) return;
    if (!permission.isGranted) {
      setState(() {
        _error = permission == CameraPermissionResult.permanentlyDenied
            ? 'Camera permission is permanently denied. Enable it in system settings.'
            : 'Camera permission was denied. Attendance needs the camera.';
        _state = AttendanceFlowState.error;
      });
      return;
    }
    try {
      final cameras = await availableCameras();
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.high,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _controller = controller;
        _state = AttendanceFlowState.idle;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _state = AttendanceFlowState.error;
      });
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _captureAndProcess() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (_photoPaths.length >= 2) return;

    setState(() {
      _state = AttendanceFlowState.capturing;
      _stageText = 'Capturing...';
    });

    try {
      final file = await controller.takePicture();
      await _processPhoto(file.path);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _state = AttendanceFlowState.error;
      });
    }
  }

  Future<void> _processPhoto(String path) async {
    final services = AppServices.instance;

    setState(() {
      _state = AttendanceFlowState.preparing;
      _stageText = 'Preparing image...';
    });

    try {
      setState(() {
        _state = AttendanceFlowState.detecting;
        _stageText = 'Finding faces...';
      });

      setState(() {
        _state = AttendanceFlowState.recognizing;
        _stageText = 'Recognizing faces...';
      });

      final results = await services.recognition.recognizeGroupImage(
        path,
        matchThreshold: PipelineConfig.matchThreshold,
        reviewThreshold: PipelineConfig.reviewThreshold,
        minMargin: PipelineConfig.minMargin,
        onStage: (stage) {
          if (!mounted) return;
          setState(() {
            switch (stage) {
              case RecognitionStage.preparing:
                _state = AttendanceFlowState.preparing;
                _stageText = 'Preparing image...';
              case RecognitionStage.detecting:
                _state = AttendanceFlowState.detecting;
                _stageText = 'Finding faces...';
              case RecognitionStage.recognizing:
                _state = AttendanceFlowState.recognizing;
                _stageText = 'Recognizing faces...';
              case RecognitionStage.merging:
                _state = AttendanceFlowState.merging;
                _stageText = 'Merging results...';
            }
          });
        },
      );

      _photoPaths.add(path);
      _photoResults.add(results);

      setState(() {
        _state = AttendanceFlowState.merging;
        _stageText = 'Merging results...';
      });

      if (_photoPaths.length < 2 && _wantsSecondPhoto(results)) {
        setState(() {
          _state = AttendanceFlowState.capturing;
          _stageText =
              'Photo ${_photoPaths.length} done (${_matchCount(results)} recognized). Capture a second photo for coverage.';
        });
        return;
      }

      setState(() => _state = AttendanceFlowState.reviewRequired);
      await _handOffToReview();
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = e.toString();
        _state = AttendanceFlowState.error;
      });
    }
  }

  bool _wantsSecondPhoto(List<GroupFaceResult> results) {
    // Offer the second photo unless everything is already high-confidence.
    return results.isEmpty ||
        results.any((r) => r.status != RecognitionStatus.match);
  }

  int _matchCount(List<GroupFaceResult> results) =>
      results.where((r) => r.status == RecognitionStatus.match).length;

  Future<void> _handOffToReview() async {
    setState(() => _state = AttendanceFlowState.reviewRequired);
    final confirmed = await Navigator.of(context).push<bool>(
      MaterialPageRoute(
        builder: (_) => ReviewFlow(
          args: widget.args,
          roster: _roster,
          photoResults: _photoResults,
        ),
      ),
    );
    if (!mounted) return;
    if (confirmed == true) {
      setState(() => _state = AttendanceFlowState.completed);
      Navigator.of(context).pop(true);
    } else {
      // Teacher aborted or not finalized: stay in capture for retry.
      setState(() {
        _state = AttendanceFlowState.capturing;
        _stageText = 'Session not saved. Retake photos or go back.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final busy = _state == AttendanceFlowState.preparing ||
        _state == AttendanceFlowState.detecting ||
        _state == AttendanceFlowState.recognizing ||
        _state == AttendanceFlowState.merging ||
        _state == AttendanceFlowState.saving;

    return PopScope(
      canPop: !busy,
      child: Scaffold(
        appBar: AppBar(
          title:
              Text('Attendance — ${widget.args.className} / ${widget.args.subjectName}'),
        ),
        body: Column(
          children: [
            Expanded(
              child: _error != null
                  ? _ErrorView(error: _error!, onRetry: _startCamera)
                  : controller != null && controller.value.isInitialized
                      ? CameraPreview(controller)
                      : const Center(child: CircularProgressIndicator()),
            ),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (_stageText.isNotEmpty)
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text(_stageText),
                    ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('Photo ${_photoPaths.length + 1} of 2'),
                      const SizedBox(width: 16),
                      if (_photoPaths.isNotEmpty)
                        TextButton(
                          onPressed: busy ? null : _handOffToReview,
                          child: const Text('Review now'),
                        ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  FilledButton.icon(
                    icon: const Icon(Icons.camera),
                    label: Text(
                      _photoPaths.isEmpty ? 'Capture photo' : 'Add photo ${_photoPaths.length + 1}',
                    ),
                    onPressed: busy || _twoPhotosDone || _state == AttendanceFlowState.reviewRequired
                        ? null
                        : _captureAndProcess,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.error, required this.onRetry});

  final String error;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, size: 48, color: Colors.red),
            const SizedBox(height: 12),
            const Text('Processing failed. Attendance was not changed.'),
            const SizedBox(height: 8),
            Text(error, style: const TextStyle(fontSize: 12)),
            const SizedBox(height: 16),
            FilledButton(onPressed: onRetry, child: const Text('Retry')),
          ],
        ),
      ),
    );
  }
}

/// Review handoff + save flow used from AttendanceScreen.
class ReviewFlow extends StatelessWidget {
  const ReviewFlow({
    super.key,
    required this.args,
    required this.roster,
    required this.photoResults,
  });

  final AttendanceArgs args;
  final List<Student> roster;
  final List<List<GroupFaceResult>> photoResults;

  @override
  Widget build(BuildContext context) {
    return ReviewScreen(
      args: ReviewArgs(
        classId: args.classId,
        className: args.className,
        subjectId: args.subjectId,
        subjectName: args.subjectName,
        roster: roster,
        photoResults: photoResults,
      ),
    );
  }
}
