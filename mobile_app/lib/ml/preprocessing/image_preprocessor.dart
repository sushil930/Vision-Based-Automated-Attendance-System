import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:image/image.dart' as img;

import '../../core/constants/pipeline_config.dart';

/// Single shared preprocessing service (plan Section 12).
///
/// Used identically by enrollment AND attendance: EXIF orientation fix,
/// longest-edge downscale (never upscale), padded face crop (desktop parity:
/// 30% x-pad, 40% y-pad), optional eye-line alignment, resize to model input,
/// and model normalization ((px - 127.5) / 127.5, RGB).
class ImagePreprocessor {
  const ImagePreprocessor();

  /// Decode + EXIF-orient + downscale a full image for detection.
  Future<PreparedImage> prepareFullImage(String path) async {
    final bytes = await File(path).readAsBytes();
    var decoded = img.decodeImage(bytes);
    if (decoded == null) {
      throw const ImageDecodeException();
    }
    final longest = math.max(decoded.width, decoded.height);
    if (longest > PipelineConfig.maxImageSize) {
      final scale = PipelineConfig.maxImageSize / longest;
      decoded = img.copyResize(
        decoded,
        width: (decoded.width * scale).round(),
        height: (decoded.height * scale).round(),
        interpolation: img.Interpolation.average,
      );
    }
    final uiImage = await _toUiImage(decoded);
    return PreparedImage(
      image: uiImage,
      width: uiImage.width,
      height: uiImage.height,
    );
  }

  /// Crop one detected face with desktop-parity padding. When both eye
  /// landmarks are available the crop is rotated so the eye line is level.
  Future<ui.Image> cropFace(
    PreparedImage source, {
    required double left,
    required double top,
    required double right,
    required double bottom,
    double? eyeLeftDx,
    double? eyeLeftDy,
    double? eyeRightDx,
    double? eyeRightDy,
  }) async {
    final w = right - left;
    final h = bottom - top;
    final padX = (w * 0.30).round();
    final padY = (h * 0.40).round();

    final leftPx = (left.round() - padX).clamp(0, source.width);
    final topPx = (top.round() - padY).clamp(0, source.height);
    final rightPx = (right.round() + padX).clamp(0, source.width);
    final bottomPx = (bottom.round() + padY).clamp(0, source.height);
    final cropW = rightPx - leftPx;
    final cropH = bottomPx - topPx;
    if (cropW <= 0 || cropH <= 0) {
      throw const ImageDecodeException('Degenerate face crop');
    }

    img.Image cropped = await _cropRaw(source, leftPx, topPx, cropW, cropH);

    final canAlign = eyeLeftDx != null &&
        eyeLeftDy != null &&
        eyeRightDx != null &&
        eyeRightDy != null;
    if (canAlign) {
      final angleRad =
          math.atan2(eyeRightDy - eyeLeftDy, eyeRightDx - eyeLeftDx);
      final angleDeg = angleRad * 180 / math.pi;
      if (angleDeg.abs() >= 1.0) {
        cropped = img.copyRotate(cropped, angle: -angleDeg);
      }
    }
    return _toUiImage(cropped);
  }

  /// Convert a face crop to the recognizer input tensor:
  /// NHWC float32 [1][size][size][3] with (px - 127.5) / 127.5.
  Future<Float32List> toModelInput(ui.Image faceCrop, int size) async {
    final resized = _resizeSync(faceCrop, size, size);
    final rgba = await resized.toByteData(format: ui.ImageByteFormat.rawRgba);
    resized.dispose();
    if (rgba == null) {
      throw const ImageDecodeException('Could not rasterize face crop');
    }
    final pixels = rgba.buffer.asUint8List();
    final input = Float32List(1 * size * size * 3);
    int out = 0;
    for (int p = 0; p < pixels.length; p += 4) {
      input[out++] = (pixels[p] - 127.5) / 127.5; // R
      input[out++] = (pixels[p + 1] - 127.5) / 127.5; // G
      input[out++] = (pixels[p + 2] - 127.5) / 127.5; // B
    }
    return input;
  }

  void dispose(PreparedImage image) => image.image.dispose();
}

class PreparedImage {
  const PreparedImage({
    required this.image,
    required this.width,
    required this.height,
  });

  final ui.Image image;
  final int width;
  final int height;
}

Future<ui.Image> _toUiImage(img.Image decoded) async {
  final png = img.encodePng(decoded);
  final buffer = await ui.ImmutableBuffer.fromUint8List(png);
  final descriptor = ui.ImageDescriptor.raw(
    buffer,
    width: decoded.width,
    height: decoded.height,
    pixelFormat: ui.PixelFormat.rgba8888,
  );
  final codec = await descriptor.instantiateCodec();
  final frame = await codec.getNextFrame();
  codec.dispose();
  descriptor.dispose();
  buffer.dispose();
  return frame.image;
}

Future<img.Image> _cropRaw(
  PreparedImage source,
  int left,
  int top,
  int w,
  int h,
) async {
  final bytes =
      await source.image.toByteData(format: ui.ImageByteFormat.rawRgba);
  if (bytes == null) {
    throw const ImageDecodeException('Could not read source pixels');
  }
  final rgba = img.Image.fromBytes(
    width: source.width,
    height: source.height,
    bytes: bytes.buffer,
    numChannels: 4,
    order: img.ChannelOrder.rgba,
  );
  return img.copyCrop(rgba, x: left, y: top, width: w, height: h);
}

ui.Image _resizeSync(ui.Image src, int w, int h) {
  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  canvas.drawImageRect(
    src,
    ui.Rect.fromLTWH(0, 0, src.width.toDouble(), src.height.toDouble()),
    ui.Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()),
    ui.Paint()..filterQuality = ui.FilterQuality.medium,
  );
  final picture = recorder.endRecording();
  final out = picture.toImageSync(w, h);
  picture.dispose();
  return out;
}

class ImageDecodeException implements Exception {
  const ImageDecodeException([this.message = 'Image could not be decoded']);
  final String message;

  @override
  String toString() => 'ImageDecodeException: $message';
}
