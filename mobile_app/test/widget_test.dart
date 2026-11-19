import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:attendance_app/features/onboarding/onboarding_screen.dart';

void main() {
  testWidgets('onboarding renders privacy contract', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: OnboardingScreen()));
    expect(find.text('Works offline'), findsOneWidget);
    expect(find.text('Internet is optional'), findsOneWidget);
  });
}
