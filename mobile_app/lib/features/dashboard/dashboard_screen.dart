import 'package:flutter/material.dart';

import '../../app/app.dart';
import '../../app/services.dart';
import '../../domain/entities/entities.dart';

/// Dashboard (Section 19): today's classes, recent attendance, quick actions.
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  List<ClassEntity> _classes = [];
  List<AttendanceSession> _recent = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final services = AppServices.instance;
    final classes = await services.classes.listAll();
    final recent =
        await services.attendance.listSessions(date: DateTime.now());
    if (!mounted) return;
    setState(() {
      _classes = classes;
      _recent = recent;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Attendance'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.of(context).pushNamed(Routes.settings),
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _load,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    icon: const Icon(Icons.class_),
                    label: const Text('Classes'),
                    onPressed: () =>
                        Navigator.of(context).pushNamed(Routes.classes),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.history),
                    label: const Text('History'),
                    onPressed: () =>
                        Navigator.of(context).pushNamed(Routes.history),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Text("Today's classes", style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (_classes.isEmpty)
              const Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Text(
                    'No classes yet. Create a class to begin.',
                  ),
                ),
              )
            else
              ..._classes.map(
                (c) => Card(
                  child: ListTile(
                    leading: const Icon(Icons.school),
                    title: Text(c.name),
                    subtitle: Text(
                        [c.section, c.semester, c.academicYear]
                            .whereType<String>()
                            .join(' · '),
                    ),
                    onTap: () => Navigator.of(context)
                        .pushNamed(Routes.students, arguments: c.id),
                  ),
                ),
              ),
            const SizedBox(height: 24),
            Text('Recent attendance',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (_recent.isEmpty)
              const Card(
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Text('No attendance taken today.'),
                ),
              )
            else
              ..._recent.map(
                (s) => Card(
                  child: ListTile(
                    leading: const Icon(Icons.fact_check),
                    title: Text('Session ${s.id.substring(0, 8)}'),
                    subtitle: Text(
                        'Status: ${s.status.name} · ${s.date.toLocal()}'),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
