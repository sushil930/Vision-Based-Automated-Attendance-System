import 'package:flutter/material.dart';

import '../../app/args.dart';
import '../../domain/entities/entities.dart';

/// Results screen (Section 19): present/absent/review/unknown counts.
class ResultsScreen extends StatelessWidget {
  const ResultsScreen({super.key, required this.args});

  final ResultsArgs args;

  @override
  Widget build(BuildContext context) {
    int present = 0, absent = 0, review = 0;
    for (final r in args.records) {
      switch (r.status) {
        case AttendanceStatus.present:
          present++;
        case AttendanceStatus.absent:
          absent++;
        case AttendanceStatus.review:
          review++;
      }
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Attendance saved')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Text(
                    '${args.className} · ${args.subjectName}',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    args.session.date.toLocal().toString().split(' ').first,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _count(context, '$present', 'Present', Colors.green),
              _count(context, '$absent', 'Absent', Colors.red),
              _count(context, '$review', 'Review', Colors.orange),
            ],
          ),
          const SizedBox(height: 24),
          ...args.records.map((r) {
            final color = switch (r.status) {
              AttendanceStatus.present => Colors.green,
              AttendanceStatus.absent => Colors.red,
              AttendanceStatus.review => Colors.orange,
            };
            return ListTile(
              leading: Icon(Icons.circle, color: color, size: 12),
              title: Text(r.studentId),
              trailing: Text(r.status.name.toUpperCase()),
              subtitle: r.confidence != null
                  ? Text('score ${r.confidence!.toStringAsFixed(3)}')
                  : null,
            );
          }),
        ],
      ),
    );
  }

  Widget _count(BuildContext context, String value, String label, Color color) {
    return Column(
      children: [
        Text(value,
            style: Theme.of(context)
                .textTheme
                .headlineLarge
                ?.copyWith(color: color)),
        Text(label),
      ],
    );
  }
}
