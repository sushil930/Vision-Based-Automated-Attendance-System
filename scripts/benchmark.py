"""Benchmark pipeline configurations over the same set of test images.

Runs each configuration (model + det_size) through identical images with
warm-up, multiple repetitions, and per-stage timing. Reports latency
statistics, detected-face counts, and — when ground-truth labels are
available (gallery/crops named face_NN mapped via profiles.json) —
recognition accuracy per configuration.

Usage:
    python scripts/benchmark.py                       # all default configs
    python scripts/benchmark.py --runs 3 --limit 10   # fewer runs/images
    python scripts/benchmark.py --configs l640,s640   # subset of configs

The gallery is never modified by this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (  # noqa: E402
    MATCH_THRESHOLD,
    MIN_MARGIN,
    get_model_for,
    load_gallery_index,
    preprocess_image,
    recognize_faces_batch,
)

LOGGER = logging.getLogger("benchmark")

# Configurations under comparison. The first is the production baseline.
BENCHMARK_CONFIGS: dict[str, tuple[str, tuple[int, int]]] = {
    "l640": ("buffalo_l", (640, 640)),
    "s640": ("buffalo_s", (640, 640)),
    "s320": ("buffalo_s", (320, 320)),
}


def load_test_images(limit: int | None) -> list[tuple[str, np.ndarray]]:
    """Load gallery crops as the shared benchmark image set (BGR)."""
    crop_dir = PROJECT_ROOT / "gallery" / "crops"
    paths = sorted(crop_dir.glob("*.jpg")) + sorted(crop_dir.glob("*.png"))
    if limit:
        paths = paths[:limit]
    images: list[tuple[str, np.ndarray]] = []
    for path in paths:
        data = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if decoded is not None:
            images.append((path.name, decoded))
    return images


def load_ground_truth() -> dict[str, str]:
    """Map crop file name -> expected student_id from the registry."""
    profiles_file = PROJECT_ROOT / "gallery" / "profiles.json"
    if not profiles_file.is_file():
        return {}
    with profiles_file.open("r", encoding="utf-8") as handle:
        profiles = json.load(handle)
    return {
        f"{key}.jpg": str(profile["student_id"])
        for key, profile in profiles.items()
        if isinstance(profile, dict) and "student_id" in profile
    }


def run_config(
    label: str,
    model_name: str,
    det_size: tuple[int, int],
    images: list[tuple[str, np.ndarray]],
    ground_truth: dict[str, str],
    runs: int,
) -> dict[str, object]:
    """Time one configuration over all images; returns aggregate stats.

    get_model_for() loads AND warms up the model (one dummy inference), so
    model startup is never charged to measured requests.
    """
    model = get_model_for(model_name, det_size)
    gallery_index = load_gallery_index()

    stage_ms = {"preprocess": 0.0, "detection": 0.0, "matching": 0.0, "annotation": 0.0}
    per_image_ms: list[float] = []
    face_counts: list[int] = []
    correct = false_match = review = unknown = missed = 0

    # Warm-up pass over the real images (not measured).
    for _name, image_bgr in images:
        resized, _scale = preprocess_image(image_bgr)
        model.get(resized)

    for _run in range(runs):
        for name, image_bgr in images:
            t0 = time.perf_counter()

            stage0 = time.perf_counter()
            resized, _scale = preprocess_image(image_bgr)
            stage_ms["preprocess"] += (time.perf_counter() - stage0) * 1000.0

            stage0 = time.perf_counter()
            faces = model.get(resized)
            stage_ms["detection"] += (time.perf_counter() - stage0) * 1000.0
            face_counts.append(len(faces))

            stage0 = time.perf_counter()
            results = (
                recognize_faces_batch([f.embedding for f in faces], gallery_index)
                if faces
                else []
            )
            stage_ms["matching"] += (time.perf_counter() - stage0) * 1000.0

            stage0 = time.perf_counter()
            annotated = resized.copy()
            for result, face in zip(results, faces):
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 2)
            stage_ms["annotation"] += (time.perf_counter() - stage0) * 1000.0

            per_image_ms.append((time.perf_counter() - t0) * 1000.0)

            # Accuracy accounting against ground truth (one expected id per crop).
            expected = ground_truth.get(name)
            if expected is None:
                continue
            if not results:
                missed += 1
                continue
            top = results[0]
            if top["status"] == "MATCH":
                if top["student_id"] == expected:
                    correct += 1
                else:
                    false_match += 1
            elif top["status"] == "REVIEW":
                review += 1
            else:
                unknown += 1

    n = max(len(per_image_ms), 1)
    images_n = max(len(images) * runs, 1)
    return {
        "label": label,
        "mean_ms": sum(per_image_ms) / n,
        "median_ms": statistics.median(per_image_ms),
        "min_ms": min(per_image_ms),
        "max_ms": max(per_image_ms),
        "avg_faces": sum(face_counts) / images_n,
        "stages": {k: v / images_n for k, v in stage_ms.items()},
        "labeled": correct + false_match + review + unknown + missed,
        "correct": correct,
        "false_match": false_match,
        "review": review,
        "unknown": unknown,
        "missed": missed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2, help="Measured passes per config")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of images")
    parser.add_argument(
        "--configs",
        type=str,
        default="l640,s640,s320",
        help="Comma-separated subset: l640,s640,s320",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    images = load_test_images(args.limit)
    if not images:
        print("No test images found in gallery/crops.")
        return
    ground_truth = load_ground_truth()

    print(f"Test images: {len(images)} | measured runs per config: {args.runs}")
    print()

    header = (
        f"{'Configuration':<20} {'Mean':>9} {'Median':>9} {'Min':>9} {'Max':>9} "
        f"{'Faces':>6} {'OK':>5} {'False':>6} {'Rev':>5} {'Unk':>5} {'Miss':>5}"
    )
    print(header)
    print("-" * len(header))

    for key in args.configs.split(","):
        key = key.strip()
        if key not in BENCHMARK_CONFIGS:
            print(f"skipping unknown config '{key}'")
            continue
        model_name, det_size = BENCHMARK_CONFIGS[key]
        stats = run_config(
            key, model_name, det_size, images, ground_truth, args.runs
        )
        print(
            f"{model_name + ' ' + str(det_size[0]) + 'x' + str(det_size[1]):<20} "
            f"{stats['mean_ms'] / 1000:>8.2f}s "
            f"{stats['median_ms'] / 1000:>8.2f}s "
            f"{stats['min_ms'] / 1000:>8.2f}s "
            f"{stats['max_ms'] / 1000:>8.2f}s "
            f"{stats['avg_faces']:>6.1f} "
            f"{stats['correct']:>5} {stats['false_match']:>6} "
            f"{stats['review']:>5} {stats['unknown']:>5} {stats['missed']:>5}"
        )
        stages = stats["stages"]
        print(
            f"{'  stages/img':<20} "
            f"preprocess={stages['preprocess']:.1f}ms  "
            f"detection={stages['detection']:.0f}ms  "
            f"matching={stages['matching']:.2f}ms  "
            f"annotation={stages['annotation']:.1f}ms"
        )

    print()
    print("Gallery was not modified. Ground truth: gallery/profiles.json "
          f"({len(ground_truth)} labeled crops).")


if __name__ == "__main__":
    main()
