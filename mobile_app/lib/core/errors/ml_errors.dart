/// Typed ML/pipeline errors (Section 43 error handling). Each maps to an
/// explicit user-facing message; none of them fabricates attendance.
sealed class MlError implements Exception {
  const MlError(this.message);
  final String message;

  @override
  String toString() =>
      '${runtimeType.toString().replaceAll('Exception', '')}: $message';
}

class CameraFailure extends MlError {
  const CameraFailure(super.message);
}

class ModelLoadFailure extends MlError {
  const ModelLoadFailure(super.message);
}

class InferenceFailure extends MlError {
  const InferenceFailure(super.message);
}

class DatabaseFailure extends MlError {
  const DatabaseFailure(super.message);
}

class OutOfMemoryFailure extends MlError {
  const OutOfMemoryFailure(super.message);
}
