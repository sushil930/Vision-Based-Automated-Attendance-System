import 'package:flutter_test/flutter_test.dart';
import 'package:attendance_app/ml/matcher/gallery_index.dart';
import 'package:attendance_app/ml/matcher/match_result.dart';

const thresholds = (
  match: 0.55,
  review: 0.45,
  margin: 0.05,
);

/// Unit vector along [axis] with optional alternating jitter to simulate noise.
List<double> axisVector(int dim, int axis, [double jitter = 0.0]) {
  final v = List<double>.filled(dim, 0.0);
  for (int c = 0; c < dim; c++) {
    v[c] = jitter * (c % 2 == 0 ? 1 : -1);
  }
  v[axis] = 1.0;
  return v;
}

MatchResult matchOne(
  GalleryIndex index,
  List<double> query,
) {
  return index.matchBatch(
    [query],
    matchThreshold: thresholds.match,
    reviewThreshold: thresholds.review,
    minMargin: thresholds.margin,
  ).single;
}

void main() {
  group('GalleryIndex construction', () {
    test('normalizes embeddings to unit length', () {
      final index = GalleryIndex(
        studentIds: ['101'],
        names: ['Alice'],
        embeddings: [
          [
            [3.0, 4.0], // norm 5 -> normalized [0.6, 0.8]
          ],
        ],
      );
      final result = matchOne(index, [0.6, 0.8]);
      expect(result.score, closeTo(1.0, 1e-6));
      expect(result.status, RecognitionStatus.match);
    });

    test('empty gallery yields UNKNOWN for every query', () {
      final index = GalleryIndex(
        studentIds: [],
        names: [],
        embeddings: [],
      );
      final result = matchOne(index, [1.0, 0.0, 0.0, 0.0]);
      expect(result.status, RecognitionStatus.unknown);
      expect(result.studentId, '-');
    });
  });

  group('classification thresholds (desktop parity)', () {
    late GalleryIndex index;

    setUp(() {
      index = GalleryIndex(
        studentIds: ['101', '102'],
        names: ['Alice', 'Bob'],
        embeddings: [
          [axisVector(8, 0)],
          [axisVector(8, 1)],
        ],
      );
    });

    test('high score + high margin -> MATCH', () {
      final r = matchOne(index, axisVector(8, 0));
      expect(r.studentId, '101');
      expect(r.status, RecognitionStatus.match);
    });

    test('mid score with near-zero margin -> not MATCH', () {
      // Equal similarity to both students -> margin ~0.
      final mixed = List<double>.filled(8, 0.5);
      final r = matchOne(index, mixed);
      expect(r.margin, lessThan(thresholds.margin));
      expect(r.status, isNot(RecognitionStatus.match));
    });

    test('low score -> UNKNOWN', () {
      final ortho = List<double>.filled(8, 0.0)..[7] = 1.0;
      final r = matchOne(index, ortho);
      expect(r.status, RecognitionStatus.unknown);
    });

    test('zero query vector -> UNKNOWN, never crashes', () {
      final r = matchOne(index, List<double>.filled(8, 0.0));
      expect(r.status, RecognitionStatus.unknown);
    });
  });

  group('per-student best sample + margin', () {
    test('best sample per student wins, margin uses runner-up student', () {
      final index = GalleryIndex(
        studentIds: ['101', '102'],
        names: ['Alice', 'Bob'],
        embeddings: [
          [axisVector(8, 0), axisVector(8, 0, 0.05)], // two Alice samples
          [axisVector(8, 1)],
        ],
      );
      // Query slightly noisy version of Alice's second sample.
      final r = matchOne(index, axisVector(8, 0, 0.04));
      expect(r.studentId, '101');
      expect(r.status, RecognitionStatus.match);
      // Margin must be vs Bob (~0 similarity), not vs Alice's own sample.
      expect(r.margin, greaterThan(0.5));
    });

    test('two identical students -> margin ~0 -> REVIEW, not MATCH', () {
      final index = GalleryIndex(
        studentIds: ['101', '102'],
        names: ['Alice', 'Bob'],
        embeddings: [
          [axisVector(8, 0)],
          [axisVector(8, 0)], // duplicate embedding, different id
        ],
      );
      final r = matchOne(index, axisVector(8, 0));
      expect(r.score, closeTo(1.0, 1e-6));
      expect(r.margin, closeTo(0.0, 1e-6));
      expect(r.status, RecognitionStatus.review);
    });
  });

  group('batch behavior', () {
    test('batch of queries maps one result per query row', () {
      final index = GalleryIndex(
        studentIds: ['101'],
        names: ['Alice'],
        embeddings: [
          [axisVector(4, 0)],
        ],
      );
      final results = index.matchBatch(
        [axisVector(4, 0), axisVector(4, 1)],
        matchThreshold: thresholds.match,
        reviewThreshold: thresholds.review,
        minMargin: thresholds.margin,
      );
      expect(results, hasLength(2));
      expect(results[0].status, RecognitionStatus.match);
      expect(results[1].status, RecognitionStatus.unknown);
    });

    test('empty query list returns empty results', () {
      final index = GalleryIndex(
        studentIds: ['101'],
        names: ['Alice'],
        embeddings: [
          [axisVector(4, 0)],
        ],
      );
      expect(index.matchBatch([], matchThreshold: 0.5, reviewThreshold: 0.4, minMargin: 0.05), isEmpty);
    });
  });
}
