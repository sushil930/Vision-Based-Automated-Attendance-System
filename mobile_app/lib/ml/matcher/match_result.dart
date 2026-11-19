/// Recognition decision outcome for a single detected face.
enum RecognitionStatus { match, review, unknown }

/// One classification result for one detected face, mirroring the desktop
/// result contract {student_id, name, score, margin, status}.
class MatchResult {
  const MatchResult({
    required this.studentId,
    required this.name,
    required this.score,
    required this.margin,
    required this.status,
  });

  final String studentId;
  final String name;
  final double score;
  final double margin;
  final RecognitionStatus status;

  Map<String, dynamic> toMap() => {
        'student_id': studentId,
        'name': name,
        'score': score,
        'margin': margin,
        'status': status.name.toUpperCase(),
      };
}
