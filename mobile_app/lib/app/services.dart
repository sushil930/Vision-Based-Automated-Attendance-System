import '../data/db/database.dart';
import '../data/repositories/attendance_repository.dart';
import '../data/repositories/repositories.dart';
import '../domain/services/backup_service.dart';
import '../domain/services/export_service.dart';
import '../ml/face_recognition_service.dart';
import '../ml/gallery_index_service.dart';
import '../core/constants/pipeline_config.dart';

/// Simple lazy service locator for v1 (Section 46 service list). Screens
/// resolve services from here; ML init happens once at startup.
class AppServices {
  AppServices._();

  static AppServices? _instance;

  static AppServices get instance => _instance ??= AppServices._();

  static const String modelId = PipelineConfig.modelId;
  static const String modelVersion = PipelineConfig.modelVersion;

  late final AppDatabase appDb;
  late final ClassRepository classes;
  late final SubjectRepository subjects;
  late final StudentRepository students;
  late final EmbeddingStore embeddings;
  late final AttendanceRepository attendance;
  late final GalleryIndexService galleryService;
  late final ModelManager modelManager;
  late final FaceRecognitionService recognition;
  late final ExportService exportService;
  late final BackupService backupService;

  Future<void> initialize() async {
    appDb = await AppDatabase.instance();
    classes = ClassRepository(appDb);
    subjects = SubjectRepository(appDb);
    students = StudentRepository(appDb);
    embeddings = EmbeddingStore(appDb);
    attendance = AttendanceRepository(appDb);
    galleryService = GalleryIndexService(embeddings);
    modelManager = ModelManager();
    recognition = FaceRecognitionService(modelManager: modelManager);
    exportService = const ExportService();
    backupService = BackupService(appDb);

    // Crash recovery before any UI (Section 52).
    await attendance.recoverStaleSessions();
  }

  /// Refreshes the in-memory gallery after enrollment changes (Section 49).
  Future<void> refreshGallery() async {
    galleryService.invalidate();
    final index = await galleryService.load();
    recognition.gallery = index;
  }
}
