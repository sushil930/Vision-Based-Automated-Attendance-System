import 'package:flutter/material.dart';

import '../../app/services.dart';
import '../../domain/entities/entities.dart';

/// History (Section 19): filter by class / subject / date.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<ClassEntity> _classes = [];
  List<AttendanceSession> _sessions = [];
  String? _classFilter;
  DateTime? _dateFilter;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final services = AppServices.instance;
    final classes = await services.classes.listAll();
    final sessions = await services.attendance.listSessions(
      classId: _classFilter,
      date: _dateFilter,
    );
    if (!mounted) return;
    setState(() {
      _classes = classes;
      _sessions = sessions;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String?>(
                    initialValue: _classFilter,
                    decoration: const InputDecoration(
                      labelText: 'Class',
                      border: OutlineInputBorder(),
                    ),
                    items: [
                      const DropdownMenuItem(value: null, child: Text('All')),
                      ..._classes.map(
                        (c) => DropdownMenuItem(value: c.id, child: Text(c.name)),
                      ),
                    ],
                    onChanged: (v) {
                      setState(() => _classFilter = v);
                      _load();
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.calendar_today, size: 18),
                    label: Text(_dateFilter == null
                        ? 'Any date'
                        : _dateFilter!.toLocal().toString().split(' ').first),
                    onPressed: () async {
                      final picked = await showDatePicker(
                        context: context,
                        initialDate: DateTime.now(),
                        firstDate: DateTime(2024),
                        lastDate: DateTime(2100),
                      );
                      if (picked != null) {
                        setState(() => _dateFilter = picked);
                        _load();
                      }
                    },
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: _sessions.isEmpty
                ? const Center(child: Text('No sessions found.'))
                : ListView.builder(
                    itemCount: _sessions.length,
                    itemBuilder: (context, i) {
                      final s = _sessions[i];
                      return ListTile(
                        leading: Icon(
                          s.status == SessionStatus.completed
                              ? Icons.check_circle
                              : Icons.error_outline,
                          color: s.status == SessionStatus.completed
                              ? Colors.green
                              : Colors.orange,
                        ),
                        title: Text('Session ${s.id.substring(0, 8)}'),
                        subtitle: Text(
                            '${s.date.toLocal().toString().split(' ').first} · ${s.status.name}'),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
