import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../../app/services.dart';
import '../../core/constants/pipeline_config.dart';

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

  Future<void> _shareAppData(BuildContext context) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final dbFile = File('${dir.path}/attendance.db');
      if (!dbFile.existsSync()) {
        if (!context.mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No database file to export yet')));
        return;
      }
      await SharePlus.instance.share(
        ShareParams(files: [XFile(dbFile.path)]),
      );
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Export failed: $e')));
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
