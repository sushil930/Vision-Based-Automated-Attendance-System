allprojects {
    repositories {
        // Mirrors first: direct Maven Central times out on this network.
        maven("https://maven.aliyun.com/repository/central")
        maven("https://maven.aliyun.com/repository/google")
        maven("https://maven.aliyun.com/repository/gradle-plugin")
        google()
        mavenCentral()
    }
}

// Align Kotlin's JVM target with the Java target (1.8) used by some plugins
// (e.g. tflite_flutter). Without this, Kotlin 2.x compiles to the JDK's 21
// while Java tasks target 1.8 -> "Inconsistent JVM Target Compatibility".
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.0")
    }
}

subprojects {
    // Align each subproject's Kotlin JVM target with its own Java target.
    // Plugins differ (tflite_flutter -> 1.8, camera_android_camerax -> 17),
    // so a single global value cannot work; read each Android extension's
    // compileOptions after evaluation.
    afterEvaluate {
        tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
            val androidExt =
                project.extensions.findByType<com.android.build.gradle.BaseExtension>()
            val javaTarget =
                androidExt?.compileOptions?.targetCompatibility?.toString() ?: "1.8"
            compilerOptions {
                jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.fromTarget(javaTarget))
            }
        }
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
