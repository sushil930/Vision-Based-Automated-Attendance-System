import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'app/app.dart';
import 'app/services.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Orientation is normalized in the image pipeline (Section 41); the app
  // itself supports both orientations for group photos.
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);

  runApp(const Bootstrap());
}

/// Splash / startup (Section 19): initialize database, model manager,
/// gallery index, configuration — once — then enter the app.
class Bootstrap extends StatefulWidget {
  const Bootstrap({super.key});

  @override
  State<Bootstrap> createState() => _BootstrapState();
}

class _BootstrapState extends State<Bootstrap> {
  String? _error;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      await AppServices.instance.initialize();
      // Model load + one-time warmup (Sections 19, 25). Failure surfaces an
      // explicit error; attendance is never fabricated without models.
      await AppServices.instance.modelManager.initialize();
      await AppServices.instance.refreshGallery();
      final prefs = await SharedPreferences.getInstance();
      final showOnboarding = !prefs.containsKey('onboarding_done');
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => AttendanceApp(showOnboarding: showOnboarding),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      setState(() => _error = error.toString());
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Attendance',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1565C0)),
      ),
      home: Scaffold(
        body: Center(
          child: _error == null
              ? const Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Loading offline AI...'),
                  ],
                )
              : Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.error_outline,
                          size: 48, color: Colors.red),
                      const SizedBox(height: 12),
                      const Text('Failed to load models'),
                      const SizedBox(height: 8),
                      Text(
                        _error!,
                        style: const TextStyle(fontSize: 12),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 16),
                      FilledButton(
                        onPressed: () {
                          setState(() => _error = null);
                          _initialize();
                        },
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                ),
        ),
      ),
    );
  }
}
