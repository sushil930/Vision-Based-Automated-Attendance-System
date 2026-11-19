import 'package:flutter_test/flutter_test.dart';
import 'package:attendance_app/domain/entities/entities.dart';
import 'package:attendance_app/domain/services/attendance_engine.dart';
import 'package:attendance_app/ml/face_recognition_service.dart';
import 'package:attendance_app/ml/matcher/match_result.dart';

DetectedFace fakeFace() =>
    const DetectedFace(left: 0, top: 0, right: 10, bottom: 10);

GroupFaceResult matchFace(String studentId, double score, double margin) =>
    GroupFaceResult(
      faceIndex: 0,
      bbox: fakeFace(),
      status: RecognitionStatus.match,
      score: score,
      margin: margin,
      studentId: studentId,
      studentName: 'Student $studentId',
    );

GroupFaceResult reviewCandidate(String studentId, int faceIndex) =>
    GroupFaceResult(
      faceIndex: faceIndex,
      bbox: fakeFace(),
      status: RecognitionStatus.review,
      score: 0.5,
      margin: 0.02,
      studentId: studentId,
      studentName: 'Student $studentId',
    );

GroupFaceResult unknownFace(int faceIndex) => GroupFaceResult(
      faceIndex: faceIndex,
      bbox: fakeFace(),
      status: RecognitionStatus.unknown,
      score: 0.1,
      margin: 0.0,
    );

List<Student> roster() => [
      student('101', '1'),
      student('102', '2'),
      student('103', '3'),
    ];

Student student(String id, String roll) => Student(
      id: id,
      classId: 'c1',
      rollNumber: roll,
      name: 'Student $id',
    );

void main() {
  const engine = AttendanceEngine();

  group('two-photo merge (Sections 17, 40)', () {
    test('same student in two photos -> one record, strongest evidence', () {
      final evidence = engine.mergePhotos([
        [matchFace('101', 0.62, 0.10)],
        [matchFace('101', 0.81, 0.20)],
      ]);
      expect(evidence, hasLength(1));
      expect(evidence['101']!.bestScore, 0.81);
      expect(evidence['101']!.sourcePhoto, 1);
    });

    test('union of photos: student in either photo becomes evidence', () {
      final evidence = engine.mergePhotos([
        [matchFace('101', 0.7, 0.2)],
        [matchFace('102', 0.66, 0.1)],
      ]);
      expect(evidence.keys, containsAll(['101', '102']));
    });

    test('UNKNOWN faces never produce evidence (Section 16)', () {
      final evidence = engine.mergePhotos([
        [unknownFace(0)],
      ]);
      expect(evidence, isEmpty);
    });
  });

  group('attendance rules (Section 44)', () {
    test('MATCH -> Present; no-evidence students stay REVIEW', () {
      final evidence = engine.mergePhotos([
        [matchFace('101', 0.9, 0.3)],
      ]);
      final records = engine.computeAttendance(
        registeredStudents: roster(),
        evidence: evidence,
        reviewDecisions: {},
        unmatchedFaces: [],
      );
      final counts = engine.countRecords(records);
      final byId = {for (final r in records) r.studentId: r};
      expect(byId['101']!.status, AttendanceStatus.present);
      expect(counts.present, 1);
      // 102/103 have no evidence, no teacher decision -> REVIEW.
      expect(counts.review, 2);
      // Absent only computed after confirmed present decisions (Section 44).
      expect(counts.absent, 0);
    });

    test('teacher-confirmed review decides present/absent', () {
      final records = engine.computeAttendance(
        registeredStudents: roster(),
        evidence: {},
        reviewDecisions: {
          '102': AttendanceStatus.present,
          '103': AttendanceStatus.absent,
        },
        unmatchedFaces: const [],
      );
      final byId = {for (final r in records) r.studentId: r};
      expect(byId['101']!.status, AttendanceStatus.review); // unresolved
      expect(byId['102']!.status, AttendanceStatus.present);
      expect(byId['102']!.reviewStatus, ReviewStatus.confirmedPresent);
      expect(byId['102']!.manuallyModified, isTrue);
      expect(byId['103']!.status, AttendanceStatus.absent);
      expect(byId['103']!.reviewStatus, ReviewStatus.confirmedAbsent);
    });

    test('ML failure never marks anyone absent (Sections 43/44)', () {
      final records = engine.computeAttendance(
        registeredStudents: roster(),
        evidence: {},
        reviewDecisions: {},
        unmatchedFaces: const [],
        mlFailed: true,
      );
      expect(records, hasLength(3));
      expect(
        records.every((r) => r.status == AttendanceStatus.review),
        isTrue,
      );
    });

    test('unresolved review stays review; unknown never becomes present', () {
      final records = engine.computeAttendance(
        registeredStudents: roster(),
        evidence: engine.mergePhotos([
          [unknownFace(0), reviewCandidate('999', 1)],
        ]),
        reviewDecisions: {},
        unmatchedFaces: [],
      );
      expect(records.every((r) => r.status == AttendanceStatus.review), isTrue);
      expect(
        records.every((r) => !r.manuallyModified),
        isTrue,
      );
    });
  });

  group('counts (Section 19)', () {
    test('counts each status correctly', () {
      final records = [
        AttendanceRecord(
            id: 'a', sessionId: 's', studentId: '1', status: AttendanceStatus.present),
        AttendanceRecord(
            id: 'b', sessionId: 's', studentId: '2', status: AttendanceStatus.absent),
        AttendanceRecord(
            id: 'c', sessionId: 's', studentId: '3', status: AttendanceStatus.review),
      ];
      final counts = engine.countRecords(records);
      expect(counts.present, 1);
      expect(counts.absent, 1);
      expect(counts.review, 1);
    });
  });
}
