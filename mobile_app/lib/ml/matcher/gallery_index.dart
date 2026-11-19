import 'dart:math' as math;
import 'dart:typed_data';

import 'match_result.dart';

/// Dart port of the desktop `GalleryIndex` (app.py).
///
/// All embeddings are L2-normalized once at load time and concatenated into a
/// single contiguous Float32List laid out as (totalRows x dimension). Row
/// offsets track where each student's block starts, so per-student best scores
/// are computed with block-max scans instead of nested per-pair loops.
///
/// Cosine similarity between unit vectors equals their dot product, so no
/// per-pair norm computation happens at query time — matching the desktop
/// algorithm exactly while keeping everything allocation-light.
class GalleryIndex {
  GalleryIndex({
    required List<String> studentIds,
    required List<String> names,
    required List<List<List<double>>> embeddings,
  })  : _ids = List.of(studentIds),
        _names = List.of(names) {
    assert(studentIds.length == names.length);
    assert(studentIds.length == embeddings.length);

    if (studentIds.isEmpty) {
      _dimension = 0;
      _matrix = Float32List(0);
      _starts = Int32List(0);
      isEmpty = true;
      return;
    }

    _dimension = embeddings.first.first.length;
    _starts = Int32List(studentIds.length);

    int totalRows = 0;
    for (final studentSamples in embeddings) {
      totalRows += studentSamples.length;
    }
    _matrix = Float32List(totalRows * _dimension);

    int cursor = 0;
    for (int i = 0; i < embeddings.length; i++) {
      _starts[i] = cursor;
      for (final vector in embeddings[i]) {
        assert(vector.length == _dimension,
            'All embeddings must share one dimension');
        final norm = _norm(vector);
        final base = cursor * _dimension;
        if (norm == 0.0) {
          for (int c = 0; c < _dimension; c++) {
            _matrix[base + c] = 0.0;
          }
        } else {
          for (int c = 0; c < _dimension; c++) {
            _matrix[base + c] = vector[c] / norm;
          }
        }
        cursor++;
      }
    }
    isEmpty = false;
  }

  final List<String> _ids;
  final List<String> _names;
  late final Float32List _matrix;
  late final Int32List _starts;
  late final int _dimension;
  late final bool isEmpty;

  int get studentCount => _ids.length;
  int get dimension => _dimension;
  List<String> get ids => List.unmodifiable(_ids);

  double _norm(List<double> v) {
    var sum = 0.0;
    for (final x in v) {
      sum += x * x;
    }
    return sum <= 0.0 ? 0.0 : math.sqrt(sum);
  }

  /// Classify every query embedding against this gallery.
  ///
  /// [queries] holds one embedding vector per detected face. Mirrors
  /// `recognize_faces_batch` from the desktop pipeline: one pass over
  /// similarities, per-student block max, margin vs runner-up.
  List<MatchResult> matchBatch(
    List<List<double>> queries, {
    required double matchThreshold,
    required double reviewThreshold,
    required double minMargin,
  }) {
    if (isEmpty || queries.isEmpty) {
      return List.generate(queries.length, (_) => _unknownResult);
    }

    final results = List<MatchResult>.filled(queries.length, _unknownResult);

    final row = Float32List(_dimension);
    for (int q = 0; q < queries.length; q++) {
      final query = queries[q];
      if (query.length != _dimension) {
        results[q] = _unknownResult;
        continue;
      }
      var norm = 0.0;
      for (int c = 0; c < _dimension; c++) {
        final v = query[c];
        row[c] = v;
        norm += v * v;
      }
      if (norm == 0.0) {
        results[q] = _unknownResult;
        continue;
      }
      norm = math.sqrt(norm);
      for (int c = 0; c < _dimension; c++) {
        row[c] /= norm;
      }

      // Per-student best score via block scan over the pre-normalized matrix.
      var bestIdx = -1;
      var bestScore = -2.0;
      var secondScore = -2.0;
      for (int s = 0; s < _ids.length; s++) {
        final start = _starts[s];
        final end = s + 1 < _ids.length ? _starts[s + 1] : cursorRow;
        var personBest = -2.0;
        for (int r = start; r < end; r++) {
          final base = r * _dimension;
          var dot = 0.0;
          for (int c = 0; c < _dimension; c++) {
            dot += row[c] * _matrix[base + c];
          }
          if (dot > personBest) {
            personBest = dot;
          }
        }
        if (personBest > bestScore) {
          secondScore = bestScore;
          bestScore = personBest;
          bestIdx = s;
        } else if (personBest > secondScore) {
          secondScore = personBest;
        }
      }

      final margin =
          bestIdx >= 0 && secondScore > -2.0 ? bestScore - secondScore : 1.0;
      results[q] = _classify(
        studentId: _ids[bestIdx],
        name: _names[bestIdx],
        score: bestScore,
        margin: margin,
        matchThreshold: matchThreshold,
        reviewThreshold: reviewThreshold,
        minMargin: minMargin,
      );
    }
    return results;
  }

  int get cursorRow => _matrix.length ~/ (_dimension == 0 ? 1 : _dimension);

  static const MatchResult _unknownResult = MatchResult(
    studentId: '-',
    name: 'Unknown',
    score: -1.0,
    margin: 1.0,
    status: RecognitionStatus.unknown,
  );

  /// Mirrors desktop `_classify_scores`.
  static MatchResult _classify({
    required String studentId,
    required String name,
    required double score,
    required double margin,
    required double matchThreshold,
    required double reviewThreshold,
    required double minMargin,
  }) {
    RecognitionStatus status;
    if (score >= matchThreshold && margin >= minMargin) {
      status = RecognitionStatus.match;
    } else if (score >= reviewThreshold) {
      status = RecognitionStatus.review;
    } else {
      status = RecognitionStatus.unknown;
    }
    return MatchResult(
      studentId: studentId,
      name: name,
      score: score,
      margin: margin,
      status: status,
    );
  }
}
