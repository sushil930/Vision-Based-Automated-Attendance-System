import 'package:flutter/material.dart';

import '../features/attendance/attendance_screen.dart';
import '../features/attendance/attendance_results_screen.dart';
import '../features/classes/classes_screen.dart';
import '../features/dashboard/dashboard_screen.dart';
import '../features/enrollment/enrollment_screen.dart';
import '../features/history/history_screen.dart';
import '../features/onboarding/onboarding_screen.dart';
import '../features/review/review_screen.dart';
import '../features/settings/settings_screen.dart';
import '../features/students/students_screen.dart';
import 'args.dart';

/// App routes (Section 7 app/router).
class Routes {
  static const onboarding = '/onboarding';
  static const dashboard = '/';
  static const classes = '/classes';
  static const classForm = '/classes/new';
  static const students = '/students';
  static const enrollment = '/enroll';
  static const attendance = '/attendance';
  static const review = '/review';
  static const results = '/results';
  static const history = '/history';
  static const settings = '/settings';
}

class AttendanceApp extends StatelessWidget {
  const AttendanceApp({super.key, required this.showOnboarding});

  final bool showOnboarding;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Attendance',
      debugShowCheckedModeBanner: false,
      theme: _theme(),
      initialRoute: showOnboarding ? Routes.onboarding : Routes.dashboard,
      routes: {
        Routes.onboarding: (_) => const OnboardingScreen(),
        Routes.dashboard: (_) => const DashboardScreen(),
        Routes.classes: (_) => const ClassesScreen(),
        Routes.classForm: (_) => const ClassFormScreen(),
        Routes.settings: (_) => const SettingsScreen(),
        Routes.history: (_) => const HistoryScreen(),
      },
      onGenerateRoute: (settings) {
        switch (settings.name) {
          case Routes.students:
            final classId = settings.arguments as String;
            return MaterialPageRoute(
              builder: (_) => StudentsScreen(classId: classId),
            );
          case Routes.enrollment:
            final args = settings.arguments as EnrollmentArgs;
            return MaterialPageRoute(
              builder: (_) => EnrollmentScreen(args: args),
            );
          case Routes.attendance:
            final args = settings.arguments as AttendanceArgs;
            return MaterialPageRoute(
              builder: (_) => AttendanceScreen(args: args),
            );
          case Routes.review:
            final args = settings.arguments as ReviewArgs;
            return MaterialPageRoute(
              builder: (_) => ReviewScreen(args: args),
            );
          case Routes.results:
            final args = settings.arguments as ResultsArgs;
            return MaterialPageRoute(
              builder: (_) => ResultsScreen(args: args),
            );
        }
        return null;
      },
    );
  }

  ThemeData _theme() {
    final base = ColorScheme.fromSeed(
      seedColor: const Color(0xFF1565C0),
      brightness: Brightness.light,
    );
    return ThemeData(
      useMaterial3: true,
      colorScheme: base,
      appBarTheme: const AppBarTheme(centerTitle: true),
    );
  }
}
