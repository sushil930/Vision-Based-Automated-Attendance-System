import 'package:attendance_app/ml/face_recognition_service.dart';

import '../../core/constants/pipeline_config.dart';
import '../../ml/matcher/match_result.dart';
import '../entities/entities.dart';

/// Two-photo merge evidence per student (Section 17).
class RecognitionEvidence {
  RecognitionEvidence({
    required this.studentId,
    required this.bestScore,
    required this.bestMargin,
    required this.sourcePhoto,
  });

  final String studentId;
  double bestScore;
  double bestMargin;
  int sourcePhoto; // 0-based photo index
}

/// Attendance decision engine (Sections 17, 44, 50).
///
/// Rules:
/// - MATCH -> Present candidate
/// - teacher-confirmed REVIEW -> Present or Absent
/// - UNKNOWN -> not Present
/// - unresolved REVIEW -> stays REVIEW
/// - ML error -> retry/review, NEVER auto-absent
/// - Absent = Registered - Confirmed Present (computed last)
class AttendanceEngine {
  const AttendanceEngine();

  /// Merge recognition results from 1–2 photos by student id, keeping the
  /// strongest valid evidence (Section 17: keep strongest evidence;
  /// Section 40: dedupe to exactly one record per student).
  Map<String, RecognitionEvidence> mergePhotos(
    List<List<GroupFaceResult>> photoResults,
  ) {
    final evidence = <String, RecognitionEvidence>{};
    for (int photo = 0; photo < photoResults.length; photo++) {
      for (final face in photoResults[photo]) {
        if (face.status != RecognitionStatus.match) {
          continue; // only MATCHes carry evidence toward Present
        }
        final id = face.studentId;
        if (id == null) continue;
        final existing = evidence[id];
        if (existing == null) {
          evidence[id] = RecognitionEvidence(
            studentId: id,
            bestScore: face.score,
            bestMargin: face.margin,
            sourcePhoto: photo,
          );
        } else if (face.score > existing.bestScore) {
          existing
            ..bestScore = face.score
            ..bestMargin = face.margin
            ..sourcePhoto = photo;
        }
      }
    }
    return evidence;
  }

  /// Build per-student records for the registered roster.
  ///
  /// [registeredStudents] is the class roster; [reviewDecisions] maps
  /// studentId -> teacher decision for REVIEW candidates; [mlFailed] marks a
  /// processing error, which must NOT mark anyone absent (Section 44).
  List<AttendanceRecord> computeAttendance({
    required List<Student> registeredStudents,
    required Map<String, RecognitionEvidence> evidence,
    required Map<String, AttendanceStatus> reviewDecisions,
    required List<GroupFaceResult> unmatchedFaces,
    bool mlFailed = false,
  }) {
    final records = <AttendanceRecord>[];
    final now = DateTime.now();

    if (mlFailed) {
      // Never fabricate attendance on ML failure (Section 43/44): everyone
      // stays in REVIEW for retry/teacher decision.
      return [
        for (final s in registeredStudents)
          AttendanceRecord(
            id: '${s.id}-pending',
            sessionId: '',
            studentId: s.id,
            status: AttendanceStatus.review,
            reviewStatus: ReviewStatus.unresolved,
            updatedAt: now,
          ),
      ];
    }

    for (final student in registeredStudents) {
      final ev = evidence[student.id];
      final teacherDecision = reviewDecisions[student.id];

      AttendanceStatus status;
      ReviewStatus reviewStatus;
      double? confidence;

      if (ev != null) {
        // High-confidence MATCH -> Present candidate.
        status = AttendanceStatus.present;
        reviewStatus = ReviewStatus.unresolved;
        confidence = ev.bestScore;
      } else if (teacherDecision != null) {
        // Teacher resolved a REVIEW/UNKNOWN candidate manually.
        status = teacherDecision;
        reviewStatus = teacherDecision == AttendanceStatus.present
            ? ReviewStatus.confirmedPresent
            : teacherDecision == AttendanceStatus.absent
                ? ReviewStatus.confirmedAbsent
                : ReviewStatus.markedUnknown;
      } else {
        status = AttendanceStatus.review;
        reviewStatus = ReviewStatus.unresolved;
      }

      records.add(AttendanceRecord(
        id: '${student.id}-rec',
        sessionId: '',
        studentId: student.id,
        status: status,
        confidence: confidence,
        sourceImageCount: photoCountWithStudent(evidence, student.id),
        manuallyModified: teacherDecision != null,
        reviewStatus: reviewStatus,
        updatedAt: now,
      ));
    }

    // Absent = Registered - Confirmed Present (Section 44). Computed last,
    // only from confirmed present decisions.
    _applyAbsentFromRoster(records);

    // Unrecognized faces stay UNKNOWN records; they never become Present.
    // (Their audit trail lives in recognition_events, not attendance rows.)
    return records;
  }

  int photoCountWithStudent(
      Map<String, RecognitionEvidence> evidence, String studentId) {
    return evidence.containsKey(studentId) ? 1 : 0;
  }

  void _applyAbsentFromRoster(List<AttendanceRecord> records) {
    // After review resolution, every non-present student is ABSENT unless the
    // review is still unresolved (then REVIEW stays REVIEW).
    for (final r in records) {
      if (r.status == AttendanceStatus.review &&
          r.reviewStatus == ReviewStatus.confirmedAbsent) {
        r.status = AttendanceStatus.absent;
      }
    }
  }

  /// Final counts for the results screen (Section 19 Results).
  AttendanceCounts countRecords(List<AttendanceRecord> records) {
    final counts = AttendanceCounts();
    for (final r in records) {
      switch (r.status) {
        case AttendanceStatus.present:
          counts.present++;
        case AttendanceStatus.absent:
          counts.absent++;
        case AttendanceStatus.review:
          counts.review++;
      }
    }
    return counts;
  }
}

class AttendanceCounts {
  int present = 0;
  int absent = 0;
  int review = 0;
  int get unknown => 0; // UNKNOWN faces never enter attendance rows

  @override
  String toString() => 'present=$present absent=$absent review=$review';
}

/// Classification helper used by the review flow to surface desktop parity.
RecognitionStatus statusFromScore({
  required double score,
  required double margin,
}) {
  if (score >= PipelineConfig.matchThreshold &&
      margin >= PipelineConfig.minMargin) {
    return RecognitionStatus.match;
  }
  if (score >= PipelineConfig.reviewThreshold) {
    return RecognitionStatus.review;
  }
  return RecognitionStatus.unknown;
}
