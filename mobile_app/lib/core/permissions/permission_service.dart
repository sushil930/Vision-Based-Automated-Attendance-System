import 'package:permission_handler/permission_handler.dart';

/// Runtime permission flow (Section 29). Requests only the camera.
/// Gallery import uses the Android Photo Picker (image_picker) which needs
/// no storage permission on modern Android versions.
enum CameraPermissionResult {
  granted,
  denied,
  permanentlyDenied;

  bool get isGranted => this == granted;
  bool get isPermanentlyDenied => this == permanentlyDenied;
}

class PermissionService {
  const PermissionService();

  /// Returns granted/denied/permanentlyDenied. A permanent denial surfaces a
  /// hint so the teacher can fix it in system settings.
  Future<CameraPermissionResult> ensureCamera() async {
    var status = await Permission.camera.status;
    if (status.isGranted) {
      return CameraPermissionResult.granted;
    }
    await Permission.camera.request();
    status = await Permission.camera.status;
    if (status.isGranted) {
      return CameraPermissionResult.granted;
    }
    if (status.isPermanentlyDenied || status.isRestricted) {
      return CameraPermissionResult.permanentlyDenied;
    }
    return CameraPermissionResult.denied;
  }

  /// Opens the app settings page when permission is permanently denied.
  Future<void> openSettings() => openAppSettings();
}
