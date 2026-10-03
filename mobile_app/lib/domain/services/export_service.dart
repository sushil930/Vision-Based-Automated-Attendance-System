import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../entities/entities.dart';

/// One flattened export row (Section 32 CSV shape).
class ExportRow {
  const ExportRow({
    required this.date,
    required this.className,
    required this.subjectName,
    required this.rollNumber,
    required this.studentName,
    required this.status,
    required this.confidence,
  });

  final DateTime date;
  final String className;
  final String subjectName;
  final String rollNumber;
  final String studentName;
  final AttendanceStatus status;
  final double? confidence;

  Map<String, Object?> toJson() => {
        'date': dateLabel,
        'class': className,
        'subject': subjectName,
        'roll_number': rollNumber,
        'student': studentName,
        'status': status.name.toUpperCase(),
        'confidence': confidence,
      };

  String get dateLabel => '${date.year.toString().padLeft(4, '0')}-'
      '${date.month.toString().padLeft(2, '0')}-'
      '${date.day.toString().padLeft(2, '0')}';
}

/// Local export (Section 32): CSV + JSON via the Android share mechanism.
/// Export must never require a server.
class ExportService {
  const ExportService();

  /// Flattens sessions + records + lookups into export rows (completed
  /// sessions only).
  List<ExportRow> buildRows({
    required List<AttendanceSession> sessions,
    required List<AttendanceRecord> records,
    required Map<String, Student> students,
    required Map<String, String> classNames,
    required Map<String, String> subjectNames,
  }) {
    final completed = {
      for (final s in sessions)
        if (s.status == SessionStatus.completed) s.id: s,
    };
    final rows = <ExportRow>[];
    for (final r in records) {
      final session = completed[r.sessionId];
      if (session == null) continue;
      final student = students[r.studentId];
      rows.add(ExportRow(
        date: session.date,
        className: classNames[session.classId] ?? session.classId,
        subjectName: subjectNames[session.subjectId] ?? session.subjectId,
        rollNumber: student?.rollNumber ?? '',
        studentName: student?.name ?? r.studentId,
        status: r.status,
        confidence: r.confidence,
      ));
    }
    rows.sort((a, b) => b.date.compareTo(a.date));
    return rows;
  }

  String toCsv(List<ExportRow> rows) {
    final buffer = StringBuffer()
      ..writeln('date,class,subject,roll_number,student,status,confidence');
    for (final r in rows) {
      buffer.writeln([
        _field(r.dateLabel),
        _field(r.className),
        _field(r.subjectName),
        _field(r.rollNumber),
        _field(r.studentName),
        _field(r.status.name.toUpperCase()),
        r.confidence?.toStringAsFixed(3) ?? '',
      ].join(','));
    }
    return buffer.toString();
  }

  String toJson(List<ExportRow> rows) {
    final payload = {
      'backup_version': 1,
      'app_version': '1.0.0',
      'export_type': 'attendance',
      'created_at': DateTime.now().toIso8601String(),
      'rows': [for (final r in rows) r.toJson()],
    };
    return const JsonEncoder.withIndent('  ').convert(payload);
  }

  Future<File> writeTemp(String name, String content) async {
    final dir = await getTemporaryDirectory();
    final out = File('${dir.path}/$name');
    await out.writeAsString(content, flush: true);
    return out;
  }

  Future<void> share(File file) async {
    await SharePlus.instance.share(ShareParams(files: [XFile(file.path)]));
  }

  static String _field(String value) {
    if (value.contains(',') ||
        value.contains('"') ||
        value.contains('\n')) {
      return '"${value.replaceAll('"', '""')}"';
    }
    return value;
  }
}
