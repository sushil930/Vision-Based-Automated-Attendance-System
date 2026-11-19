/// Domain entities (plan Section 8 data model). Kept free of storage details.
library;

class Teacher {
  Teacher({required this.id, required this.name, this.createdAt, this.updatedAt});

  final String id;
  final String name;
  final DateTime? createdAt;
  final DateTime? updatedAt;
}

class ClassEntity {
  ClassEntity({
    required this.id,
    required this.name,
    this.section,
    this.semester,
    this.academicYear,
    this.createdAt,
    this.updatedAt,
  });

  final String id;
  final String name;
  final String? section;
  final String? semester;
  final String? academicYear;
  final DateTime? createdAt;
  final DateTime? updatedAt;
}

class Subject {
  Subject({required this.id, required this.name, this.code, this.createdAt, this.updatedAt});

  final String id;
  final String name;
  final String? code;
  final DateTime? createdAt;
  final DateTime? updatedAt;
}

class ClassSubject {
  ClassSubject({required this.id, required this.classId, required this.subjectId});

  final String id;
  final String classId;
  final String subjectId;
}

enum StudentStatus { active, inactive }

class Student {
  Student({
    required this.id,
    required this.classId,
    required this.rollNumber,
    required this.name,
    this.status = StudentStatus.active,
    this.createdAt,
    this.updatedAt,
  });

  final String id;
  final String classId;
  final String rollNumber;
  final String name;
  final StudentStatus status;
  final DateTime? createdAt;
  final DateTime? updatedAt;
}

class FaceProfile {
  FaceProfile({
    required this.id,
    required this.studentId,
    required this.modelId,
    required this.modelVersion,
    required this.embeddingDimension,
    required this.preprocessingVersion,
    this.createdAt,
    this.updatedAt,
  });

  final String id;
  final String studentId;
  final String modelId;
  final String modelVersion;
  final int embeddingDimension;
  final String preprocessingVersion;
  final DateTime? createdAt;
  final DateTime? updatedAt;
}

class FaceEmbedding {
  FaceEmbedding({
    required this.id,
    required this.faceProfileId,
    required this.embedding,
    required this.sampleIndex,
    this.qualityScore,
    this.createdAt,
  });

  final String id;
  final String faceProfileId;
  final List<double> embedding;
  final int sampleIndex;
  final double? qualityScore;
  final DateTime? createdAt;
}

enum AttendanceStatus { present, absent, review }

enum SessionStatus { processing, reviewRequired, completed, error }

class AttendanceSession {
  AttendanceSession({
    required this.id,
    required this.classId,
    required this.subjectId,
    required this.date,
    this.startedAt,
    this.completedAt,
    this.status = SessionStatus.processing,
  });

  final String id;
  final String classId;
  final String subjectId;
  final DateTime date;
  final DateTime? startedAt;
  final DateTime? completedAt;
  final SessionStatus status;
}

enum ReviewStatus { unresolved, confirmedPresent, confirmedAbsent, markedUnknown }

class AttendanceRecord {
  AttendanceRecord({
    required this.id,
    required this.sessionId,
    required this.studentId,
    required this.status,
    this.confidence,
    this.sourceImageCount = 0,
    this.manuallyModified = false,
    this.reviewStatus = ReviewStatus.unresolved,
    this.updatedAt,
  });

  final String id;
  final String sessionId;
  final String studentId;
  AttendanceStatus status;
  double? confidence;
  int sourceImageCount;
  bool manuallyModified;
  ReviewStatus reviewStatus;
  DateTime? updatedAt;
}

/// Optional audit trail (Section 8). Never stores raw images/embeddings.
class RecognitionEvent {
  RecognitionEvent({
    required this.id,
    required this.sessionId,
    this.studentId,
    required this.score,
    required this.margin,
    required this.decision,
    required this.faceIndex,
    this.createdAt,
  });

  final String id;
  final String sessionId;
  final String? studentId;
  final double score;
  final double margin;
  final String decision;
  final int faceIndex;
  final DateTime? createdAt;
}
