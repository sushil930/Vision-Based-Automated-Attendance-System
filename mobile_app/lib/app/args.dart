import '../domain/entities/entities.dart';
import '../ml/face_recognition_service.dart';

/// Router argument types shared across feature screens.

class EnrollmentArgs {
  const EnrollmentArgs({required this.studentId, required this.studentName});

  final String studentId;
  final String studentName;
}

class AttendanceArgs {
  const AttendanceArgs({
    required this.classId,
    required this.className,
    required this.subjectId,
    required this.subjectName,
  });

  final String classId;
  final String className;
  final String subjectId;
  final String subjectName;
}

class ReviewArgs {
  const ReviewArgs({
    required this.classId,
    required this.className,
    required this.subjectId,
    required this.subjectName,
    required this.roster,
    required this.photoResults,
  });

  final String classId;
  final String className;
  final String subjectId;
  final String subjectName;
  final List<Student> roster;
  final List<List<GroupFaceResult>> photoResults;
}

class ResultsArgs {
  const ResultsArgs({
    required this.session,
    required this.records,
    required this.className,
    required this.subjectName,
  });

  final AttendanceSession session;
  final List<AttendanceRecord> records;
  final String className;
  final String subjectName;
}
