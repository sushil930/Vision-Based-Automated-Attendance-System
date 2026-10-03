import 'package:flutter_test/flutter_test.dart';

import 'package:attendance_app/domain/entities/entities.dart';
import 'package:attendance_app/domain/services/export_service.dart';

void main() {
  group('ExportService CSV (Section 32)', () {
    const export = ExportService();

    test('CSV header matches Section 32 spec', () {
      final csv = export.toCsv(const []);
      expect(csv, startsWith('date,class,subject,roll_number,student,status,confidence'));
    });

    test('CSV row quotes fields containing commas', () {
      final csv = export.toCsv([
        ExportRow(
          date: DateTime(2026, 10, 3),
          className: '10A',
          subjectName: 'Math',
          rollNumber: '1',
          studentName: 'Patel, Ravi',
          status: AttendanceStatus.present,
          confidence: 0.91,
        ),
      ]);
      expect(csv, contains('"Patel, Ravi"'));
      expect(csv, contains('PRESENT'));
      expect(csv, contains('0.910'));
    });

    test('JSON payload has envelope fields and rows', () {
      final json = export.toJson([
        ExportRow(
          date: DateTime(2026, 10, 3),
          className: '10A',
          subjectName: 'Math',
          rollNumber: '1',
          studentName: 'Ravi',
          status: AttendanceStatus.review,
          confidence: null,
        ),
      ]);
      expect(json, contains('"backup_version"'));
      expect(json, contains('"roll_number": "1"'));
      expect(json, contains('"status": "REVIEW"'));
      expect(json, contains('"confidence": null'));
    });

    test('buildRows filters to completed sessions only', () {
      final rows = export.buildRows(
        sessions: [
          AttendanceSession(
              id: 's1', classId: 'c1', subjectId: 'u1',
              date: DateTime(2026, 10, 3), status: SessionStatus.completed),
          AttendanceSession(
              id: 's2', classId: 'c1', subjectId: 'u1',
              date: DateTime(2026, 10, 3), status: SessionStatus.error),
        ],
        records: [
          AttendanceRecord(id: 'r1', sessionId: 's1', studentId: 'st1',
              status: AttendanceStatus.present, confidence: 0.9),
          AttendanceRecord(id: 'r2', sessionId: 's2', studentId: 'st2',
              status: AttendanceStatus.present, confidence: 0.9),
        ],
        students: {'st1': Student(
          id: 'st1', classId: 'c1', rollNumber: '1', name: 'Ravi')},
        classNames: {'c1': '10A'},
        subjectNames: {'u1': 'Math'},
      );
      expect(rows, hasLength(1));
      expect(rows.single.rollNumber, '1');
    });
  });
}
