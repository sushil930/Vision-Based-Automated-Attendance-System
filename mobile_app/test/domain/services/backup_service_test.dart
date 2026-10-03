import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

import 'package:attendance_app/core/constants/pipeline_config.dart';
import 'package:attendance_app/data/db/database.dart';
import 'package:attendance_app/data/repositories/repositories.dart';
import 'package:attendance_app/domain/entities/entities.dart';
import 'package:attendance_app/domain/services/backup_service.dart';

void main() {
  late Directory tempBackupDir;

  setUpAll(() {
    TestWidgetsFlutterBinding.ensureInitialized();
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });

  setUp(() async {
    await AppDatabase.resetForTesting();
    // Fresh temp directory per test so no DB file persists between tests.
    tempBackupDir = Directory.systemTemp.createTempSync('att_backup_');
    final dbDir = Directory.systemTemp.createTempSync('att_db_');
    await databaseFactory.setDatabasesPath(dbDir.path);
  });

  Future<AppDatabase> freshDb() => AppDatabase.instance();

  BackupService backupFor(AppDatabase db) =>
      BackupService(db, backupDirectory: tempBackupDir);

  test('createBackup produces a JSON envelope with model metadata', () async {
    final db = await freshDb();
    final backup = backupFor(db);

    final student = Student(
        id: 'st1', classId: 'c1', rollNumber: '1', name: 'Ravi');
    await StudentRepository(db).create(student);

    final file = await backup.createBackup();
    final content = await file.readAsString();
    final payload = jsonDecode(content) as Map<String, dynamic>;

    expect(payload['backup_version'], 1);
    expect(payload['model_id'], PipelineConfig.modelId);
    expect(payload['model_version'], PipelineConfig.modelVersion);
    expect(payload['embedding_dimension'], PipelineConfig.embeddingDimension);
    final data = payload['data'] as Map<String, dynamic>;
    expect((data['students'] as List), isNotEmpty);
  });

  test('restore refuses incompatible embedding space (Section 34)', () async {
    final db = await freshDb();
    final backup = backupFor(db);
    final file = await backup.createBackup();

    final content = await file.readAsString();
    final payload = jsonDecode(content) as Map<String, dynamic>;
    payload['model_id'] = 'other_model_v9';
    final badFile = File('${file.path}.bad.json');
    await badFile.writeAsString(jsonEncode(payload));

    await expectLater(
      backup.restore(badFile),
      throwsA(isA<IncompatibleBackupException>()),
    );
  });

  test('restore round-trips embeddings through the store', () async {
    final db = await freshDb();
    final backup = backupFor(db);
    final embeddings = EmbeddingStore(db);

    await StudentRepository(db).create(
      Student(id: 'st1', classId: 'c1', rollNumber: '1', name: 'Ravi'),
    );
    await embeddings.saveProfileWithEmbeddings(studentId: 'st1', samples: [
      List<double>.filled(512, 0.1),
      List<double>.filled(512, 0.2),
    ]);

    final file = await backup.createBackup();

    await embeddings.deleteForStudent('st1');
    await backup.restore(file);

    final loaded = await embeddings.loadAllForCurrentModel();
    expect(loaded.containsKey('st1'), isTrue);
    expect(loaded['st1']!.length, 2);
    expect(loaded['st1']!.first.first, closeTo(0.1, 1e-6));
  });
}
