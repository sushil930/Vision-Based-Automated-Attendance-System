import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';

import '../../app/args.dart';
import '../../app/app.dart';
import '../../app/services.dart';
import '../../domain/entities/entities.dart';
import '../../domain/services/attendance_engine.dart';

/// Teacher review (Section 18): confirm Present / Mark Absent / Unknown,
/// assign students to unknown faces, then finalize attendance.
class ReviewScreen extends StatefulWidget {
  const ReviewScreen({super.key, required this.args});

  final ReviewArgs args;

  @override
  State<ReviewScreen> createState() => _ReviewScreenState();
}

class _ReviewScreenState extends State<ReviewScreen> {
  final AttendanceEngine _engine = const AttendanceEngine();
  late final Map<String, AttendanceStatus> _reviewDecisions;
  Map<String, RecognitionEvidence> _evidence = {};
  List<AttendanceRecord> _records = const [];
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _reviewDecisions = {};
    _recompute();
  }

  void _recompute() {
    _evidence = _engine.mergePhotos(widget.args.photoResults);
    _records = _engine.computeAttendance(
      registeredStudents: widget.args.roster,
      evidence: _evidence,
      reviewDecisions: _reviewDecisions,
      unmatchedFaces: widget.args.photoResults.expand((p) => p).toList(),
    );
  }

  Future<void> _decide(String studentId, AttendanceStatus status) async {
    setState(() => _reviewDecisions[studentId] = status);
    _recompute();
  }

  Future<void> _finalize() async {
    setState(() => _saving = true);
    try {
      final services = AppServices.instance;
      final session = await services.attendance.createSession(
        id: const Uuid().v4(),
        classId: widget.args.classId,
        subjectId: widget.args.subjectId,
        date: DateTime.now(),
      );
      final records = [
        for (final r in _records)
          AttendanceRecord(
            id: '${session.id}-${r.studentId}',
            sessionId: session.id,
            studentId: r.studentId,
            status: r.status,
            confidence: r.confidence,
            sourceImageCount: r.sourceImageCount,
            manuallyModified: r.manuallyModified,
            reviewStatus: r.reviewStatus,
            updatedAt: DateTime.now(),
          ),
      ];
      await services.attendance.completeSession(
        sessionId: session.id,
        records: records,
      );
      if (!mounted) return;
      Navigator.of(context).pushReplacementNamed(
        Routes.results,
        arguments: ResultsArgs(
          session: session,
          records: records,
          className: widget.args.className,
          subjectName: widget.args.subjectName,
        ),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Save failed: $error')),
      );
      setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final counts = _engine.countRecords(_records);
    final pendingReview =
        _records.where((r) => r.status == AttendanceStatus.review).toList();

    return Scaffold(
      appBar: AppBar(title: const Text('Review attendance')),
      body: _saving
          ? const Center(child: CircularProgressIndicator())
          : ListView(
              padding: const EdgeInsets.all(16),
              children: [
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _count(context, 'Present', counts.present, Colors.green),
                        _count(context, 'Absent', counts.absent, Colors.red),
                        _count(context, 'Review', counts.review, Colors.orange),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 16),
                if (pendingReview.isEmpty)
                  const Card(
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Text('No uncertain cases. Ready to save.'),
                    ),
                  )
                else
                  ...pendingReview.map((r) {
                    final student = widget.args.roster
                        .firstWhere((s) => s.id == r.studentId);
                    return Card(
                      child: Padding(
                        padding: const EdgeInsets.all(12),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(student.name,
                                style: Theme.of(context).textTheme.titleMedium),
                            Text(
                              'Confidence: '
                              '${r.confidence?.toStringAsFixed(2) ?? "no evidence"}',
                            ),
                            const SizedBox(height: 8),
                            Row(
                              children: [
                                Expanded(
                                  child: FilledButton.tonal(
                                    onPressed: () =>
                                        _decide(r.studentId, AttendanceStatus.present),
                                    child: const Text('Confirm Present'),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: OutlinedButton(
                                    onPressed: () =>
                                        _decide(r.studentId, AttendanceStatus.absent),
                                    child: const Text('Mark Absent'),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: TextButton(
                                    onPressed: () => _decide(
                                        r.studentId, AttendanceStatus.review),
                                    child: const Text('Unknown'),
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    );
                  }),
                const SizedBox(height: 24),
                FilledButton(
                  onPressed: _finalize,
                  child: const Text('Save attendance'),
                ),
                const SizedBox(height: 24),
              ],
            ),
    );
  }

  Widget _count(BuildContext context, String label, int value, Color color) {
    return Column(
      children: [
        Text('$value',
            style: Theme.of(context)
                .textTheme
                .headlineMedium
                ?.copyWith(color: color)),
        Text(label),
      ],
    );
  }
}
