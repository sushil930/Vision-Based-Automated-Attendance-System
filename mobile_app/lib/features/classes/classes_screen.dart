import 'package:flutter/material.dart';

import '../../app/app.dart';
import '../../app/services.dart';
import '../../domain/entities/entities.dart';

/// Classes screen (Section 19): create class, open class.
class ClassesScreen extends StatefulWidget {
  const ClassesScreen({super.key});

  @override
  State<ClassesScreen> createState() => _ClassesScreenState();
}

class _ClassesScreenState extends State<ClassesScreen> {
  List<ClassEntity> _classes = [];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final classes = await AppServices.instance.classes.listAll();
    if (!mounted) return;
    setState(() => _classes = classes);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Classes')),
      floatingActionButton: FloatingActionButton.extended(
        icon: const Icon(Icons.add),
        label: const Text('New class'),
        onPressed: () async {
          await Navigator.of(context).pushNamed(Routes.classForm);
          _load();
        },
      ),
      body: _classes.isEmpty
          ? const Center(child: Text('No classes. Create one to begin.'))
          : ListView.builder(
              itemCount: _classes.length,
              itemBuilder: (context, i) {
                final c = _classes[i];
                return ListTile(
                  leading: const Icon(Icons.class_),
                  title: Text(c.name),
                  subtitle: Text(
                      [c.section, c.semester, c.academicYear]
                          .whereType<String>()
                          .join(' · ')),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () => Navigator.of(context)
                      .pushNamed(Routes.students, arguments: c.id),
                );
              },
            ),
    );
  }
}

/// Create-class form.
class ClassFormScreen extends StatefulWidget {
  const ClassFormScreen({super.key});

  @override
  State<ClassFormScreen> createState() => _ClassFormScreenState();
}

class _ClassFormScreenState extends State<ClassFormScreen> {
  final _formKey = GlobalKey<FormState>();
  final _name = TextEditingController();
  final _section = TextEditingController();
  final _semester = TextEditingController();
  final _year = TextEditingController();
  bool _saving = false;

  @override
  void dispose() {
    _name.dispose();
    _section.dispose();
    _semester.dispose();
    _year.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() => _saving = true);
    await AppServices.instance.classes.create(
      id: DateTime.now().microsecondsSinceEpoch.toString(),
      name: _name.text.trim(),
      section: _section.text.trim().isEmpty ? null : _section.text.trim(),
      semester: _semester.text.trim().isEmpty ? null : _semester.text.trim(),
      academicYear: _year.text.trim().isEmpty ? null : _year.text.trim(),
    );
    if (!mounted) return;
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('New class')),
      body: Form(
        key: _formKey,
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            TextFormField(
              controller: _name,
              decoration: const InputDecoration(labelText: 'Class name *'),
              validator: (v) =>
                  v == null || v.trim().isEmpty ? 'Name required' : null,
            ),
            TextFormField(
              controller: _section,
              decoration: const InputDecoration(labelText: 'Section'),
            ),
            TextFormField(
              controller: _semester,
              decoration: const InputDecoration(labelText: 'Semester'),
            ),
            TextFormField(
              controller: _year,
              decoration: const InputDecoration(labelText: 'Academic year'),
            ),
            const SizedBox(height: 24),
            FilledButton(
              onPressed: _saving ? null : _save,
              child: Text(_saving ? 'Saving...' : 'Create class'),
            ),
          ],
        ),
      ),
    );
  }
}
