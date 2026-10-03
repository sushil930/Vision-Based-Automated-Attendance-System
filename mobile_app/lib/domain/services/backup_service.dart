import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

import '../../core/constants/pipeline_config.dart';
import '../../data/db/database.dart';

/// Local backup/restore (Section 33): classes, subjects, students,
/// embeddings, attendance, settings. Raw enrollment photos are NOT included.
///
/// Backup envelope:
/// { backup_version, app_version, model_id, model_version, created_at, data }
///
/// Restore refuses backups whose embedding space is incompatible
/// (model_id/model_version/embedding_dimension mismatch) instead of silently
/// mixing embedding spaces (Section 34). Restore runs in one transaction
/// (Section 51).
class BackupService {
  BackupService(this._appDb, {this.backupDirectory});

  final AppDatabase _appDb;
  final Directory? backupDirectory;

  Database get db => _appDb.db;

  static const int backupVersion = 1;

  static const List<String> _tables = [
    'teachers',
    'classes',
    'subjects',
    'class_subjects',
    'students',
    'face_profiles',
    'face_embeddings',
    'attendance_sessions',
    'attendance_records',
  ];

  Future<File> createBackup() async {
    final data = <String, Object?>{};
    for (final table in _tables) {
      final rows = await db.query(table);
      data[table] = [
        for (final row in rows)
          // Blobs are not JSON-serializable; encode as base64 (Section 33).
          {
            for (final entry in row.entries)
              entry.key: entry.value is Uint8List
                  ? base64Encode(entry.value as Uint8List)
                  : entry.value,
          },
      ];
    }

    final payload = {
      'backup_version': backupVersion,
      'app_version': '1.0.0',
      'model_id': PipelineConfig.modelId,
      'model_version': PipelineConfig.modelVersion,
      'embedding_dimension': PipelineConfig.embeddingDimension,
      'preprocessing_version': PipelineConfig.preprocessingVersion,
      'created_at': DateTime.now().toIso8601String(),
      'data': data,
    };

    final dir = await backupDir();
    final file = File(
      '${dir.path}/attendance_backup_${DateTime.now().millisecondsSinceEpoch}.json',
    );
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(payload),
      flush: true,
    );
    return file;
  }

  Future<RestoreResult> restore(File backupFile) async {
    final content = await backupFile.readAsString();
    final payload = jsonDecode(content) as Map<String, dynamic>;

    final version = payload['backup_version'];
    if (version is! int || version > backupVersion) {
      throw IncompatibleBackupException('Unsupported backup version: $version');
    }
    if (payload['model_id'] != PipelineConfig.modelId ||
        payload['model_version'] != PipelineConfig.modelVersion ||
        payload['embedding_dimension'] != PipelineConfig.embeddingDimension) {
      throw const IncompatibleBackupException(
        'Backup embeddings were generated with a different model. '
        'Incompatible embeddings are refused; re-enrollment is required.',
      );
    }

    final data = payload['data'] as Map<String, dynamic>;
    await db.transaction((txn) async {
      // Children first on delete, parents first on insert.
      await txn.delete('attendance_records');
      await txn.delete('attendance_sessions');
      await txn.delete('face_embeddings');
      await txn.delete('face_profiles');
      await txn.delete('students');
      await txn.delete('class_subjects');
      await txn.delete('subjects');
      await txn.delete('classes');
      await txn.delete('teachers');

      for (final table in _tables) {
        final rows = (data[table] as List<dynamic>? ?? const []);
        for (final row in rows.cast<Map<String, dynamic>>()) {
          final decoded = <String, Object?>{};
          for (final entry in row.entries) {
            final value = entry.value;
            if (table == 'face_embeddings' &&
                entry.key == 'embedding_blob' &&
                value is String) {
              decoded[entry.key] =
                  Uint8List.fromList(base64Decode(value));
            } else if (value == null ||
                value is String ||
                value is int ||
                value is double) {
              decoded[entry.key] = value;
            }
          }
          await txn.insert(table, decoded);
        }
      }
    });

    return RestoreResult(
      classes: (data['classes'] as List? ?? const []).length,
      subjects: (data['subjects'] as List? ?? const []).length,
      students: (data['students'] as List? ?? const []).length,
      sessions: (data['attendance_sessions'] as List? ?? const []).length,
      records: (data['attendance_records'] as List? ?? const []).length,
    );
  }

  Future<Directory> backupDir() async {
    final override = backupDirectory;
    if (override != null) {
      return override;
    }
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/backups');
    if (!dir.existsSync()) {
      dir.createSync(recursive: true);
    }
    return dir;
  }
}

class RestoreResult {
  const RestoreResult({
    required this.classes,
    required this.subjects,
    required this.students,
    required this.sessions,
    required this.records,
  });

  final int classes;
  final int subjects;
  final int students;
  final int sessions;
  final int records;

  @override
  String toString() =>
      'Restored: $classes classes, $subjects subjects, $students students, '
      '$sessions sessions, $records records';
}

class IncompatibleBackupException implements Exception {
  const IncompatibleBackupException(this.message);
  final String message;

  @override
  String toString() => 'IncompatibleBackupException: $message';
}
