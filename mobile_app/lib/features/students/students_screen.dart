import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';

import '../../app/app.dart';
import '../../app/args.dart';
import '../../app/services.dart';
import '../../domain/entities/entities.dart';

/// Students screen (Section 19): roll, name, enrolled badge, registration,
/// and entry into attendance for the class.
class StudentsScreen extends StatefulWidget {
  const StudentsScreen({super.key, required this.classId});

  final String classId;

  @override
  State<StudentsScreen> createState() => _StudentsScreenState();
}

class _StudentsScreenState extends State<StudentsScreen> {
  List<Student> _students = [];
  Set<String> _enrolled = {};
  String _className = '';

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final services = AppServices.instance;
    final students = await services.students.listForClass(widget.classId);
    final enrolled = await services.embeddings.enrolledStudentIds();
    final classes = await services.classes.listAll();
    final name = classes
        .firstWhere((c) => c.id == widget.classId,
            orElse: () => ClassEntity(id: widget.classId, name: 'Class'))
        .name;
    if (!mounted) return;
    setState(() {
      _students = students;
      _enrolled = enrolled;
      _className = name;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(_className)),
      floatingActionButton: FloatingActionButton.extended(
        icon: const Icon(Icons.person_add),
        label: const Text('Register'),
        onPressed: () => _showRegisterDialog(context),
      ),
      body: _students.isEmpty
          ? const Center(
              child: Text('No students registered yet.'),
            )
          : ListView.builder(
              itemCount: _students.length,
              itemBuilder: (context, i) {
                final s = _students[i];
                final isEnrolled = _enrolled.contains(s.id);
                return ListTile(
                  leading: CircleAvatar(child: Text(s.rollNumber)),
                  title: Text(s.name),
                  subtitle: Text('Roll ${s.rollNumber}'),
                  trailing: isEnrolled
                      ? const Icon(Icons.verified, color: Colors.green)
                      : TextButton(
                          onPressed: () => Navigator.of(context).pushNamed(
                            Routes.enrollment,
                            arguments: EnrollmentArgs(
                              studentId: s.id,
                              studentName: s.name,
                            ),
                          ),
                          child: const Text('Enroll'),
                        ),
                );
              },
            ),
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: FilledButton.icon(
            icon: const Icon(Icons.photo_camera),
            label: const Text('Take attendance'),
            onPressed: _students.isEmpty
                ? null
                : () => Navigator.of(context).pushNamed(
                      Routes.attendance,
                      arguments: AttendanceArgs(
                        classId: widget.classId,
                        className: _className,
                        subjectId: 'default',
                        subjectName: 'Subject',
                      ),
                    ),
          ),
        ),
      ),
    );
  }

  void _showRegisterDialog(BuildContext context) {
    final nameCtrl = TextEditingController();
    final rollCtrl = TextEditingController();
    final formKey = GlobalKey<FormState>();

    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (sheetContext) => Padding(
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(sheetContext).viewInsets.bottom,
          left: 16,
          right: 16,
          top: 16,
        ),
        child: Form(
          key: formKey,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text('Register student',
                  style: Theme.of(sheetContext).textTheme.titleMedium),
              const SizedBox(height: 16),
              TextFormField(
                controller: nameCtrl,
                decoration: const InputDecoration(labelText: 'Name *'),
                validator: (v) =>
                    v == null || v.trim().isEmpty ? 'Required' : null,
              ),
              TextFormField(
                controller: rollCtrl,
                decoration: const InputDecoration(labelText: 'Roll number *'),
                keyboardType: TextInputType.text,
                validator: (v) =>
                    v == null || v.trim().isEmpty ? 'Required' : null,
              ),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () async {
                  if (!formKey.currentState!.validate()) return;
                  final services = AppServices.instance;
                  final student = Student(
                    id: const Uuid().v4(),
                    classId: widget.classId,
                    rollNumber: rollCtrl.text.trim(),
                    name: nameCtrl.text.trim(),
                  );
                  await services.students.create(student);
                  if (!sheetContext.mounted) return;
                  Navigator.of(sheetContext).pop();
                  // Straight into face enrollment (Section 9).
                  if (!context.mounted) return;
                  await Navigator.of(context).pushNamed(
                    Routes.enrollment,
                    arguments: EnrollmentArgs(
                      studentId: student.id,
                      studentName: student.name,
                    ),
                  );
                  _load();
                },
                child: const Text('Save & enroll face'),
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
