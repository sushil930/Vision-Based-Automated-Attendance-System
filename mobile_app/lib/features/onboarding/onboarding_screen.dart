import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../app/app.dart';

/// Onboarding (Section 19): explain offline-first privacy contract.
class OnboardingScreen extends StatelessWidget {
  const OnboardingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Icon(Icons.face_retouching_off, size: 72),
              const SizedBox(height: 24),
              Text(
                'Works offline',
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 32),
              _bullet(context, Icons.wifi_off, 'Works completely offline'),
              _bullet(context, Icons.phone_android,
                  'Face processing happens on this device'),
              _bullet(context, Icons.cloud_off, 'Internet is optional'),
              _bullet(context, Icons.delete_outline,
                  'Face data can be deleted anytime'),
              const SizedBox(height: 32),
              FilledButton(
                onPressed: () async {
                  final prefs = await SharedPreferences.getInstance();
                  await prefs.setBool('onboarding_done', true);
                  if (!context.mounted) return;
                  Navigator.of(context)
                      .pushReplacementNamed(Routes.dashboard);
                },
                child: const Text('Get started'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _bullet(BuildContext context, IconData icon, String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        children: [
          Icon(icon, size: 22),
          const SizedBox(width: 12),
          Expanded(child: Text(text)),
        ],
      ),
    );
  }
}
