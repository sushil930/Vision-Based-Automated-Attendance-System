import '../data/repositories/repositories.dart';
import 'matcher/gallery_index.dart';

/// Builds and caches the recognition gallery from stored embeddings
/// (Section 49): load, normalize once, contiguous matrix, student mapping,
/// cache, invalidate after enrollment changes, rebuild once.
///
/// Mirrors the desktop `load_gallery_index` mtime-based caching concept:
/// the index is rebuilt only when a new enrollment transaction commits,
/// never per comparison.
class GalleryIndexService {
  GalleryIndexService(this._embeddings);

  final EmbeddingStore _embeddings;
  GalleryIndex? _index;
  bool _dirty = true;
  int _version = 0;

  bool get isStale => _index == null || _dirty;

  /// Loads (or rebuilds once after invalidation) the index.
  Future<GalleryIndex> load() async {
    if (_index != null && !_dirty) {
      return _index!;
    }
    final perStudent = await _embeddings.loadAllForCurrentModel();
    final ids = perStudent.keys.toList(growable: false);
    final samples = [
      for (final id in ids) perStudent[id]!,
    ];
    final names = [
      for (final id in ids) 'Student $id',
    ];
    _index = GalleryIndex(
      studentIds: ids,
      names: names,
      embeddings: samples,
    );
    _dirty = false;
    _version++;
    return _index!;
  }

  GalleryIndex? get current => _index;

  int get version => _version;

  /// Called after any enrollment change (Section 49): invalidate once,
  /// rebuild on the next recognition request.
  void invalidate() {
    _dirty = true;
  }
}
