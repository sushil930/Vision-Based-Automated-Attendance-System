# R8 rules for release shrinking (plan T13.5).

# tflite_flutter: the LiteRT GPU delegate references an optional
# GPU API package that is not bundled with the CPU-only runtime.
# The delegate is never constructed in this app (CPU-only v1), so
# the missing class is safe to ignore.
-dontwarn org.tensorflow.lite.gpu.GpuDelegateFactory$Options

# Keep LiteRT classes reachable from JNI. The interpreter is created
# and invoked from native code via reflection-like bindings; shrinking
# would break inference at runtime.
-keep class org.tensorflow.lite.** { *; }
