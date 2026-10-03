import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import '../../app/services.dart';
import '../../core/constants/pipeline_config.dart';
import '../../domain/entities/entities.dart';
import '../../domain/services/backup_service.dart';

/// Settings (Section 19): model information, storage usage, clear face data,
/// clear attendance data, privacy, about. Thresholds are developer-only and
/// intentionally not exposed here (Section 19 final paragraph).
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  String _dbSize = '…';

  @override
  void initState() {
    super.initState();
    _measureStorage();
  }

  Future<void> _measureStorage() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final size = await Directory(dir.path)
          .list(recursive: true)
          .fold<int>(0, (sum, entity) {
        if (entity is File) return sum + entity.lengthSync();
        return sum;
      });
      if (!mounted) return;
      setState(() => _dbSize = '${(size / 1024 / 1024).toStringAsFixed(1)} MB');
    } catch (_) {
      if (!mounted) return;
      setState(() => _dbSize = 'unavailable');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        children: [
          const _Header('Model'),
          ListTile(
            leading: const Icon(Icons.memory),
            title: const Text('Recognition model'),
            subtitle: Text(
                '${PipelineConfig.modelId} v${PipelineConfig.modelVersion} · '
                '${PipelineConfig.embeddingDimension}-d · '
                'preprocess ${PipelineConfig.preprocessingVersion}'),
          ),
          const _Header('Storage'),
          ListTile(
            leading: const Icon(Icons.storage),
            title: const Text('Storage usage'),
            subtitle: Text(_dbSize),
          ),
          ListTile(
            leading: const Icon(Icons.ios_share),
            title: const Text('Export attendance (CSV/JSON)'),
            subtitle: const Text('Shares a CSV or JSON export'),
            onTap: () => _exportAttendance(context),
          ),
          ListTile(
            leading: const Icon(Icons.save_alt),
            title: const Text('Backup'),
            subtitle: const Text('Creates a versioned local backup'),
            onTap: () => _createBackup(context),
          ),
          ListTile(
            leading: const Icon(Icons.restore),
            title: const Text('Restore'),
            subtitle: const Text('Restores a versioned backup file'),
            onTap: () => _restoreBackup(context),
          ),
          const _Header('Data'),
          ListTile(
            leading: const Icon(Icons.face_retouching_off),
            title: const Text('Clear face data'),
            subtitle: const Text('Deletes all stored embeddings'),
            onTap: () => _confirmDanger(
              context,
              'Clear all face data?',
              'All enrolled face embeddings will be deleted. Students remain.',
              _clearFaceData,
            ),
          ),
          ListTile(
            leading: const Icon(Icons.fact_check_outlined),
            title: const Text('Clear attendance data'),
            subtitle: const Text('Deletes all sessions and records'),
            onTap: () => _confirmDanger(
              context,
              'Clear all attendance data?',
              'All attendance sessions and records will be deleted.',
              _clearAttendanceData,
            ),
          ),
          const _Header('Privacy'),
          const ListTile(
            leading: Icon(Icons.lock_outline),
            title: Text('On-device processing'),
            subtitle: Text(
                'Face detection and recognition run on this device. '
                'No biometric data leaves the phone. Internet is optional.'),
          ),
          const ListTile(
            leading: Icon(Icons.info_outline),
            title: Text('About'),
            subtitle: Text('AI Attendance 1.0.0 — offline classroom attendance'),
          ),
        ],
      ),
    );
  }

  /// Section 32: CSV/JSON export via Android share, offline only.
  Future<void> _exportAttendance(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final services = AppServices.instance;
      final sessions = await services.attendance.listSessions();
      final classes = {for (final c in await services.classes.listAll()) c.id: c.name};
      final subjects = {
        for (final s in await services.subjects.listAll()) s.id: s.name
      };
      final studentsById = <String, Student>{};
      for (final c in await services.classes.listAll()) {
        for (final s in await services.students.listForClass(c.id)) {
          studentsById[s.id] = s;
        }
      }
      final records = <AttendanceRecord>[];
      for (final session in sessions) {
        records.addAll(await services.attendance.recordsForSession(session.id));
      }

      final rows = services.exportService.buildRows(
        sessions: sessions,
        records: records,
        students: studentsById,
        classNames: classes,
        subjectNames: subjects,
      );
      if (rows.isEmpty) {
        messenger.showSnackBar(
            const SnackBar(content: Text('Nothing to export yet')));
        return;
      }
      final choice = await showModalBottomSheet<String>(
        context: context,
        builder: (sheetContext) => SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.table_chart),
                title: const Text('Export as CSV'),
                onTap: () => Navigator.of(sheetContext).pop('csv'),
              ),
              ListTile(
                leading: const Icon(Icons.data_object),
                title: const Text('Export as JSON'),
                onTap: () => Navigator.of(sheetContext).pop('json'),
              ),
            ],
          ),
        ),
      );
      if (choice == null || !context.mounted) return;
      final content = choice == 'csv'
          ? services.exportService.toCsv(rows)
          : services.exportService.toJson(rows);
      final file = await services.exportService.writeTemp(
        'attendance_export_${DateTime.now().millisecondsSinceEpoch}.${choice == 'csv' ? 'csv' : 'json'}',
        content,
      );
      await services.exportService.share(file);
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Export failed: $e')));
    }
  }

  /// Section 33: versioned local backup.
  Future<void> _createBackup(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final file = await AppServices.instance.backupService.createBackup();
      messenger.showSnackBar(
        SnackBar(content: Text('Backup saved: ${file.path.split(Platform.pathSeparator).last}')),
      );
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Backup failed: $e')));
    }
  }

  /// Section 33: restore with compatibility validation.
  Future<void> _restoreBackup(BuildContext context) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final dir = await AppServices.instance.backupService.backupDir();
      final files = dir
          .listSync()
          .whereType<File>()
          .where((f) => f.path.endsWith('.json'))
          .toList()
        ..sort((a, b) => b.path.compareTo(a.path));
      if (files.isEmpty) {
        messenger.showSnackBar(
            const SnackBar(content: Text('No backups found')));
        return;
      }
      final choice = await showModalBottomSheet<File>(
        context: context,
        builder: (sheetContext) => SafeArea(
          child: ListView(
            shrinkWrap: true,
            children: [
              for (final f in files)
                ListTile(
                  leading: const Icon(Icons.description),
                  title: Text(f.path.split(Platform.pathSeparator).last),
                  onTap: () => Navigator.of(sheetContext).pop(f),
                ),
            ],
          ),
        ),
      );
      if (choice == null || !context.mounted) return;
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (dialogContext) => AlertDialog(
          title: const Text('Restore backup?'),
          content: const Text(
              'This replaces all current classes, students, faces and '
              'attendance with the backup contents.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(false),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.of(dialogContext).pop(true),
              child: const Text('Restore'),
            ),
          ],
        ),
      );
      if (confirmed != true || !context.mounted) return;
      final result =
          await AppServices.instance.backupService.restore(choice);
      messenger.showSnackBar(SnackBar(content: Text(result.toString())));
    } on IncompatibleBackupException catch (e) {
      messenger.showSnackBar(SnackBar(content: Text(e.toString())));
    } catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('Restore failed: $e')));
    }
  }

  Future<void> _clearFaceData() async {
    final services = AppServices.instance;
    final classes = await services.classes.listAll();
    for (final c in classes) {
      final list = await services.students.listForClass(c.id);
      for (final s in list) {
        await services.embeddings.deleteForStudent(s.id);
      }
    }
    await services.refreshGallery();
    if (!mounted) return;
    ScaffoldMessenger.of(context)
        .showSnackBar(const SnackBar(content: Text('Face data cleared')));
  }

  Future<void> _clearAttendanceData() async {
    final db = AppServices.instance.appDb;
    await db.db.transaction((txn) async {
      await txn.delete('attendance_records');
      await txn.delete('recognition_events');
      await txn.delete('attendance_sessions');
    });
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Attendance data cleared')));
  }

  void _confirmDanger(
    BuildContext context,
    String title,
    String message,
    Future<void> Function() action,
  ) {
    showDialog<void>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () async {
              Navigator.of(dialogContext).pop();
              await action();
            },
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header(this.title);

  final String title;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
      child: Text(
        title,
        style: Theme.of(context)
            .textTheme
            .labelLarge
            ?.copyWith(color: Theme.of(context).colorScheme.primary),
      ),
    );
  }
}
