#!/usr/bin/env python3
"""Build a configurable T-Core+Other YOLO foreign-object dataset.

This file is generated in two standalone variants:

* T-Core: original foreign-object classes with at least T unique positive images
  are retained; lower-support classes are excluded from the target labels.
* T-Core+Other: classes with at least T unique positive images remain separate;
  lower-support classes are remapped, instance by instance, to ``Other``.

Thresholding and balancing are image-level operations. All retained object
instances are preserved. Polygon rows are converted to tight axis-aligned
bounding boxes for detection and are also preserved as polygons and masks.

Dependencies:
    pip install pyyaml pillow
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import re
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from PIL import Image, ImageDraw


VERSION = "3.0.0"
VARIANT = "core_other"  # "core" or "core_other"
VARIANT_TITLE = "T-Core+Other"

LOGGER = logging.getLogger("threshold_dataset_builder")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
CONTEXT_NAMES = {
    "wagon",
    "wagen",
    "collectionwagon",
    "greenwastewagon",
    "container",
    "collectioncontainer",
    "greenwastecontainer",
}


@dataclass(frozen=True)
class SourceDataset:
    name: str
    yaml_path: Path
    root: Path
    class_names: dict[int, str]
    images: tuple[Path, ...]


@dataclass
class Annotation:
    source_class_id: int
    source_class_name: str
    normalized_class_name: str
    geometry: str  # bbox | polygon
    bbox_xyxy: tuple[float, float, float, float]
    polygon: list[tuple[float, float]] | None = None
    line_number: int = 0

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


@dataclass
class ImageRecord:
    source_name: str
    source_yaml: Path
    image_path: Path
    label_path: Path
    width: int
    height: int
    sha256: str
    output_name: str
    context: str
    objects: list[Annotation] = field(default_factory=list)


@dataclass
class TargetObject:
    output_class_id: int
    output_class_name: str
    annotation: Annotation


@dataclass
class TaskImage:
    record: ImageRecord
    targets: list[TargetObject]
    ignored_foreign_objects: list[Annotation]

    @property
    def positive(self) -> bool:
        return bool(self.targets)

    @property
    def class_ids(self) -> set[int]:
        return {target.output_class_id for target in self.targets}

    @property
    def all_targets_polygon(self) -> bool:
        return bool(self.targets) and all(
            target.annotation.geometry == "polygon" for target in self.targets
        )


class DatasetError(RuntimeError):
    """Raised for data conditions that would make the output unreliable."""


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold().replace("ß", "ss")
    return re.sub(r"[^a-z0-9]+", "", text)


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("_.-") or "dataset"


def parse_yaml_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(name) for index, name in enumerate(raw)}
    if isinstance(raw, dict):
        result = {int(key): str(value) for key, value in raw.items()}
        return dict(sorted(result.items()))
    raise DatasetError("YOLO YAML field 'names' must be a list or mapping.")


def resolve_root(yaml_path: Path, raw_path: Any) -> Path:
    if raw_path in (None, ""):
        return yaml_path.parent.resolve()
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = yaml_path.parent / path
    return path.resolve()


def images_from_yaml_entry(root: Path, entry: Any) -> list[Path]:
    entries = entry if isinstance(entry, list) else [entry]
    result: list[Path] = []
    for item in entries:
        if item in (None, ""):
            continue
        path = Path(str(item)).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.is_dir():
            result.extend(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.casefold() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.casefold() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                candidate = Path(line).expanduser()
                if not candidate.is_absolute():
                    candidate = root / candidate
                candidate = candidate.resolve()
                if candidate.is_file() and candidate.suffix.casefold() in IMAGE_EXTENSIONS:
                    result.append(candidate)
        elif path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            result.append(path)
    return result


def load_source(yaml_path: Path, source_index: int) -> SourceDataset:
    """Load one source strictly from sibling images/ and labels/ folders.

    The source YAML names field defines class IDs. Its path/train/val/test values
    are intentionally ignored so copied or stale YAML paths cannot redirect the
    builder away from the July-26 source folder.
    """
    yaml_path = yaml_path.expanduser().resolve()
    if not yaml_path.is_file():
        raise DatasetError(f"YAML file does not exist: {yaml_path}")
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
    names = parse_yaml_names(raw.get("names"))
    root = yaml_path.parent.resolve()
    images_root = root / "images"
    labels_root = root / "labels"
    if not images_root.is_dir():
        raise DatasetError(f"Sibling images folder is missing: {images_root}")
    if not labels_root.is_dir():
        raise DatasetError(f"Sibling labels folder is missing: {labels_root}")
    images = sorted(
        {candidate.resolve() for candidate in images_root.rglob("*")
         if candidate.is_file() and candidate.suffix.casefold() in IMAGE_EXTENSIONS},
        key=lambda path: str(path).casefold(),
    )
    if not images:
        raise DatasetError(f"No images found under sibling folder: {images_root}")
    base_name = slugify(yaml_path.stem)
    return SourceDataset(
        name=f"{base_name}_{source_index:02d}",
        yaml_path=yaml_path,
        root=root,
        class_names=names,
        images=tuple(images),
    )


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_indexes = [
        index for index, part in enumerate(parts) if part.casefold() == "images"
    ]
    if image_indexes:
        index = image_indexes[-1]
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    parent = image_path.parent
    if (
        parent.name.casefold() in {"train", "val", "valid", "test"}
        and parent.parent.name.casefold() == "images"
    ):
        return (
            parent.parent.parent
            / "labels"
            / parent.name
            / f"{image_path.stem}.txt"
        )
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def context_from_filename(path: Path) -> str:
    stem = path.stem.casefold()
    if stem.endswith("w"):
        return "wagon"
    if stem.endswith(("c", "b", "t")):
        return "container"
    return "unknown"


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
    *,
    attempts: int = 6,
    base_delay: float = 0.5,
) -> str:
    """Hash one file, retrying transient Windows sharing/permission failures."""
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(chunk_size):
                    digest.update(chunk)
            return digest.hexdigest()
        except PermissionError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = base_delay * attempt
            LOGGER.warning(
                "Read access temporarily denied for %s; retry %d/%d in %.1f s",
                path,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)
        except OSError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = base_delay * attempt
            LOGGER.warning(
                "Temporary file-read error for %s (%s); retry %d/%d in %.1f s",
                path,
                exc,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)

    raise DatasetError(
        "Unable to read image bytes after "
        f"{attempts} attempts: {path}. Last error: {last_error}. "
        "Close programs that may hold the file, verify that the current Windows "
        "account has Read permission, and test the file independently before "
        "rerunning the create stage."
    )


def image_dimensions_with_retry(
    path: Path,
    *,
    attempts: int = 6,
    base_delay: float = 0.5,
) -> tuple[int, int]:
    """Read image dimensions with the same transient-access retry policy."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with Image.open(path) as image:
                width, height = image.size
            return int(width), int(height)
        except (PermissionError, OSError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            delay = base_delay * attempt
            LOGGER.warning(
                "Unable to open image %s (%s); retry %d/%d in %.1f s",
                path,
                exc,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)

    raise DatasetError(
        "Unable to open image after "
        f"{attempts} attempts: {path}. Last error: {last_error}."
    )


def finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def parse_label(
    label_path: Path,
    class_names: dict[int, str],
    *,
    allow_missing: bool,
    invalid_row_policy: str,
    coordinate_tolerance: float = 1e-6,
) -> tuple[list[Annotation], list[str]]:
    if not label_path.exists():
        if allow_missing:
            return [], [f"missing_label_allowed: {label_path}"]
        raise DatasetError(
            f"Missing label file: {label_path}. Use --allow-missing-labels only "
            "when missing labels are confirmed true negatives."
        )

    annotations: list[Annotation] = []
    warnings: list[str] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        error_prefix = f"{label_path}:{line_number}"
        try:
            class_value = float(parts[0])
            class_id = int(class_value)
            if class_value != class_id:
                raise ValueError("class ID is not an integer")
            if class_id not in class_names:
                raise ValueError(f"class ID {class_id} is absent from YAML names")
            values = [float(value) for value in parts[1:]]
            if not finite(values):
                raise ValueError("geometry contains NaN or infinity")

            class_name = class_names[class_id]
            normalized = normalize_name(class_name)
            if len(values) == 4:
                xc, yc, width, height = values
                if not (
                    -coordinate_tolerance <= xc <= 1 + coordinate_tolerance
                    and -coordinate_tolerance <= yc <= 1 + coordinate_tolerance
                    and 0 < width <= 1 + coordinate_tolerance
                    and 0 < height <= 1 + coordinate_tolerance
                ):
                    raise ValueError(f"invalid normalized bbox {values}")
                x1 = xc - width / 2.0
                y1 = yc - height / 2.0
                x2 = xc + width / 2.0
                y2 = yc + height / 2.0
                if (
                    x1 < -coordinate_tolerance
                    or y1 < -coordinate_tolerance
                    or x2 > 1 + coordinate_tolerance
                    or y2 > 1 + coordinate_tolerance
                ):
                    raise ValueError("bbox extends outside normalized image")
                bbox = (
                    max(0.0, x1),
                    max(0.0, y1),
                    min(1.0, x2),
                    min(1.0, y2),
                )
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError("bbox has zero or negative area")
                annotations.append(
                    Annotation(
                        class_id,
                        class_name,
                        normalized,
                        "bbox",
                        bbox,
                        None,
                        line_number,
                    )
                )
                continue

            if len(values) >= 6 and len(values) % 2 == 0:
                points = list(zip(values[0::2], values[1::2]))
                if len(points) >= 2 and points[0] == points[-1]:
                    points.pop()
                if len(points) < 3:
                    raise ValueError("polygon requires at least three vertices")
                if any(
                    not (
                        -coordinate_tolerance <= x <= 1 + coordinate_tolerance
                        and -coordinate_tolerance <= y <= 1 + coordinate_tolerance
                    )
                    for x, y in points
                ):
                    raise ValueError("polygon contains coordinates outside image")
                points = [
                    (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))
                    for x, y in points
                ]
                if polygon_area(points) <= 0:
                    raise ValueError("polygon has zero area")
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                annotations.append(
                    Annotation(
                        class_id,
                        class_name,
                        normalized,
                        "polygon",
                        (min(xs), min(ys), max(xs), max(ys)),
                        points,
                        line_number,
                    )
                )
                continue

            raise ValueError(
                "row must contain class+4 bbox values or class+at least 3 x/y pairs"
            )
        except (ValueError, IndexError) as exc:
            message = f"{error_prefix}: {exc}"
            if invalid_row_policy == "error":
                raise DatasetError(message) from exc
            warnings.append(f"skipped_invalid_row: {message}")
    return annotations, warnings


def safe_output_name(source_name: str, image_path: Path) -> str:
    return f"{slugify(source_name)}__{image_path.name}"


def scan_sources(args: argparse.Namespace) -> tuple[list[ImageRecord], list[str]]:
    sources = [load_source(path, index + 1) for index, path in enumerate(args.yaml)]
    records: list[ImageRecord] = []
    warnings: list[str] = []
    total = sum(len(source.images) for source in sources)
    processed = 0
    for source in sources:
        LOGGER.info("Scanning %s (%d images)", source.yaml_path, len(source.images))
        for image_path in source.images:
            processed += 1
            label_path = infer_label_path(image_path)
            annotations, row_warnings = parse_label(
                label_path,
                source.class_names,
                allow_missing=args.allow_missing_labels,
                invalid_row_policy=args.invalid_row_policy,
            )
            warnings.extend(row_warnings)
            width, height = image_dimensions_with_retry(
                image_path,
                attempts=args.file_read_retries,
                base_delay=args.file_read_delay,
            )
            if width <= 0 or height <= 0:
                raise DatasetError(f"Invalid image dimensions for {image_path}")
            digest = sha256_file(
                image_path,
                attempts=args.file_read_retries,
                base_delay=args.file_read_delay,
            )
            records.append(
                ImageRecord(
                    source_name=source.name,
                    source_yaml=source.yaml_path,
                    image_path=image_path,
                    label_path=label_path,
                    width=width,
                    height=height,
                    sha256=digest,
                    output_name=safe_output_name(source.name, image_path),
                    context=context_from_filename(image_path),
                    objects=annotations,
                )
            )
            if processed % 500 == 0 or processed == total:
                LOGGER.info("Scanned %d/%d images", processed, total)
    return records, warnings


def remove_or_group_duplicates(
    records: list[ImageRecord], deduplicate: bool
) -> tuple[list[ImageRecord], list[dict[str, Any]]]:
    by_hash: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)

    duplicate_rows: list[dict[str, Any]] = []
    kept: list[ImageRecord] = []
    for digest, group in by_hash.items():
        signatures = {
            tuple(
                sorted(
                    (
                        obj.normalized_class_name,
                        obj.geometry,
                        tuple(round(value, 8) for value in obj.bbox_xyxy),
                        tuple(
                            (round(x, 8), round(y, 8))
                            for x, y in (obj.polygon or [])
                        ),
                    )
                    for obj in record.objects
                )
            )
            for record in group
        }
        if len(signatures) > 1:
            paths = "; ".join(str(record.image_path) for record in group)
            raise DatasetError(
                "Exact duplicate image bytes have conflicting labels: " + paths
            )
        representative = sorted(group, key=lambda item: str(item.image_path).casefold())[0]
        kept.extend([representative] if deduplicate else group)
        for duplicate in group[1:]:
            duplicate_rows.append(
                {
                    "sha256": digest,
                    "kept_image": str(representative.image_path),
                    "duplicate_image": str(duplicate.image_path),
                    "removed": deduplicate,
                }
            )
    kept.sort(key=lambda record: str(record.image_path).casefold())
    return kept, duplicate_rows


def is_context_class(normalized_name: str) -> bool:
    return normalized_name in CONTEXT_NAMES


def original_class_statistics(
    records: Sequence[ImageRecord], threshold: int, threshold_rule: str
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    image_sets: dict[str, set[str]] = defaultdict(set)
    instance_counts: Counter[str] = Counter()
    polygon_counts: Counter[str] = Counter()
    bbox_counts: Counter[str] = Counter()
    canonical_names: dict[str, str] = {}

    for record in records:
        classes_in_image: set[str] = set()
        for obj in record.objects:
            if is_context_class(obj.normalized_class_name):
                continue
            canonical_names.setdefault(obj.normalized_class_name, obj.source_class_name)
            classes_in_image.add(obj.normalized_class_name)
            instance_counts[obj.normalized_class_name] += 1
            if obj.geometry == "polygon":
                polygon_counts[obj.normalized_class_name] += 1
            else:
                bbox_counts[obj.normalized_class_name] += 1
        for normalized_name in classes_in_image:
            image_sets[normalized_name].add(record.sha256)

    result: dict[str, dict[str, Any]] = {}
    for normalized_name in sorted(canonical_names):
        positive_images = len(image_sets[normalized_name])
        frequent = (
            positive_images >= threshold
            if threshold_rule == "at_least"
            else positive_images > threshold
        )
        result[normalized_name] = {
            "normalized_name": normalized_name,
            "class_name": canonical_names[normalized_name],
            "positive_images": positive_images,
            "instances": instance_counts[normalized_name],
            "polygon_instances": polygon_counts[normalized_name],
            "bbox_instances": bbox_counts[normalized_name],
            "frequent": frequent,
        }
    return result, canonical_names


def construct_task(
    records: Sequence[ImageRecord],
    class_stats: dict[str, dict[str, Any]],
) -> tuple[list[TaskImage], tuple[str, ...], dict[str, int], set[str]]:
    frequent = {
        normalized_name
        for normalized_name, stats in class_stats.items()
        if stats["frequent"]
    }
    rare = set(class_stats) - frequent

    if VARIANT == "core" and not frequent:
        raise DatasetError(
            "No foreign-object class passed the threshold. Lower --threshold or "
            "use T-Core+Other."
        )

    frequent_order = sorted(
        frequent,
        key=lambda name: (
            -class_stats[name]["positive_images"],
            class_stats[name]["class_name"].casefold(),
        ),
    )
    output_names = [class_stats[name]["class_name"] for name in frequent_order]
    normalized_to_output = {
        normalized_name: index for index, normalized_name in enumerate(frequent_order)
    }
    if VARIANT == "core_other" and rare:
        other_id = len(output_names)
        output_names.append("Other")
        for normalized_name in rare:
            normalized_to_output[normalized_name] = other_id

    task_images: list[TaskImage] = []
    for record in records:
        targets: list[TargetObject] = []
        ignored: list[Annotation] = []
        for obj in record.objects:
            normalized = obj.normalized_class_name
            if is_context_class(normalized):
                continue
            if normalized in normalized_to_output:
                output_id = normalized_to_output[normalized]
                output_name = output_names[output_id]
                targets.append(TargetObject(output_id, output_name, obj))
            else:
                ignored.append(obj)
        task_images.append(TaskImage(record, targets, ignored))

    return task_images, tuple(output_names), normalized_to_output, rare


def class_image_counts(records: Sequence[TaskImage], n_classes: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for record in records:
        for class_id in record.class_ids:
            counts[class_id] += 1
    for class_id in range(n_classes):
        counts[class_id] += 0
    return counts


def class_instance_counts(records: Sequence[TaskImage], n_classes: int) -> Counter[int]:
    counts = Counter(
        target.output_class_id for record in records for target in record.targets
    )
    for class_id in range(n_classes):
        counts[class_id] += 0
    return counts


def select_balanced_positives(
    positives: list[TaskImage],
    n_classes: int,
    mode: str,
    requested_target: int | None,
    seed: int,
) -> tuple[list[TaskImage], dict[str, Any]]:
    available = class_image_counts(positives, n_classes)
    if any(available[class_id] == 0 for class_id in range(n_classes)):
        missing = [class_id for class_id in range(n_classes) if available[class_id] == 0]
        raise DatasetError(f"Output classes without positive images: {missing}")

    if mode == "none":
        return positives.copy(), {
            "class_balance": "none",
            "target_positive_images_per_class": None,
            "available_positive_images_per_class": dict(available),
            "selected_positive_images_per_class": dict(
                class_image_counts(positives, n_classes)
            ),
        }

    target = min(available.values())
    if requested_target is not None:
        if requested_target <= 0:
            raise DatasetError("--class-target must be positive")
        target = min(target, requested_target)

    rng = random.Random(seed)
    candidates = positives.copy()
    rng.shuffle(candidates)
    selected: list[TaskImage] = []
    selected_ids: set[int] = set()
    counts: Counter[int] = Counter()

    while any(counts[class_id] < target for class_id in range(n_classes)):
        best_index: int | None = None
        best_score = -float("inf")
        for index, record in enumerate(candidates):
            if index in selected_ids:
                continue
            deficits = {
                class_id: max(0, target - counts[class_id])
                for class_id in record.class_ids
            }
            useful = sum(1 for deficit in deficits.values() if deficit > 0)
            if useful == 0:
                continue
            weighted_deficit = sum(
                deficit / max(available[class_id], 1)
                for class_id, deficit in deficits.items()
            )
            overshoot = sum(
                max(0, counts[class_id] + 1 - target)
                for class_id in record.class_ids
            )
            score = 1000.0 * useful + 100.0 * weighted_deficit - overshoot
            score += rng.random() * 1e-6
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            deficits = {
                class_id: target - counts[class_id]
                for class_id in range(n_classes)
                if counts[class_id] < target
            }
            raise DatasetError(
                f"Could not satisfy class-image target {target}; deficits={deficits}"
            )
        record = candidates[best_index]
        selected_ids.add(best_index)
        selected.append(record)
        for class_id in record.class_ids:
            counts[class_id] += 1

    # Remove redundant images while retaining at least target images for every class.
    order = list(range(len(selected)))
    rng.shuffle(order)
    keep = [True] * len(selected)
    for index in order:
        record = selected[index]
        if all(counts[class_id] - 1 >= target for class_id in record.class_ids):
            keep[index] = False
            for class_id in record.class_ids:
                counts[class_id] -= 1
    selected = [record for index, record in enumerate(selected) if keep[index]]

    return selected, {
        "class_balance": "multilabel_undersample",
        "target_positive_images_per_class": target,
        "available_positive_images_per_class": dict(available),
        "selected_positive_images_per_class": dict(
            class_image_counts(selected, n_classes)
        ),
    }


def proportional_stratified_sample(
    records: list[TaskImage], count: int, seed: int
) -> list[TaskImage]:
    if count >= len(records):
        return records.copy()
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[TaskImage]] = defaultdict(list)
    for record in records:
        strata[(record.record.source_name, record.record.context)].append(record)
    for values in strata.values():
        rng.shuffle(values)

    exact = {
        key: count * len(values) / len(records) for key, values in strata.items()
    }
    allocated = {key: min(len(strata[key]), int(value)) for key, value in exact.items()}
    remaining = count - sum(allocated.values())
    order = sorted(
        strata,
        key=lambda key: (exact[key] - allocated[key], rng.random()),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if allocated[key] < len(strata[key]):
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    selected = [record for key, values in strata.items() for record in values[: allocated[key]]]
    rng.shuffle(selected)
    return selected


def balance_dataset(
    task_images: list[TaskImage],
    n_classes: int,
    args: argparse.Namespace,
) -> tuple[list[TaskImage], dict[str, Any], list[TaskImage]]:
    positives = [record for record in task_images if record.positive]
    all_foreign_negative = [
        record
        for record in task_images
        if not any(
            not is_context_class(obj.normalized_class_name)
            for obj in record.record.objects
        )
    ]
    task_negative = [record for record in task_images if not record.positive]

    selected_positives, class_balance_info = select_balanced_positives(
        positives,
        n_classes,
        args.class_balance,
        args.class_target,
        args.seed,
    )

    if VARIANT == "core" and args.negative_policy == "task":
        negative_pool = task_negative
    else:
        negative_pool = all_foreign_negative

    desired_negatives = int(round(len(selected_positives) * args.negative_ratio))
    desired_negatives = max(0, desired_negatives)
    selected_negatives = proportional_stratified_sample(
        negative_pool,
        min(desired_negatives, len(negative_pool)),
        args.seed + 1009,
    )
    selected = selected_positives + selected_negatives
    random.Random(args.seed + 2017).shuffle(selected)

    excluded_rare_only = [
        record
        for record in task_images
        if not record.positive and record.ignored_foreign_objects
    ]
    info = {
        **class_balance_info,
        "available_positive_images": len(positives),
        "selected_positive_images": len(selected_positives),
        "available_clean_negative_images": len(all_foreign_negative),
        "available_task_negative_images": len(task_negative),
        "selected_negative_images": len(selected_negatives),
        "requested_negative_images": desired_negatives,
        "achieved_negative_positive_ratio": (
            len(selected_negatives) / len(selected_positives)
            if selected_positives
            else None
        ),
        "negative_policy": (
            args.negative_policy if VARIANT == "core" else "clean"
        ),
        "rare_only_images_not_used_as_clean_negatives": len(excluded_rare_only),
    }
    return selected, info, excluded_rare_only


def signature(record: TaskImage) -> str:
    classes = ",".join(str(class_id) for class_id in sorted(record.class_ids))
    classes = classes or "NEG"
    polygon_state = (
        "POLY"
        if record.all_targets_polygon
        else "MIXED_OR_BOX"
        if record.positive
        else "NEG"
    )
    return f"{classes}|{record.record.context}|{polygon_state}|{record.record.source_name}"


def integer_split_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLITS}
    targets = {split: int(math.floor(raw[split])) for split in SPLITS}
    remaining = total - sum(targets.values())
    order = sorted(
        SPLITS,
        key=lambda split: (raw[split] - targets[split], ratios[split]),
        reverse=True,
    )
    for split in order[:remaining]:
        targets[split] += 1
    return targets


def allocate_greedy(
    records: list[TaskImage],
    ratios: dict[str, float],
    n_classes: int,
    rng: random.Random,
) -> dict[str, list[TaskImage]]:
    """Capacity-aware multilabel split allocation.

    Exact duplicate hashes remain in the same split. Assignment minimizes
    incremental deviations in total images, class-positive images, positive and
    negative images, source contribution, and multilabel signatures.
    """
    groups_by_hash: dict[str, list[TaskImage]] = defaultdict(list)
    for record in records:
        groups_by_hash[record.record.sha256].append(record)
    groups = list(groups_by_hash.values())

    total_images = len(records)
    image_targets = integer_split_targets(total_images, ratios)
    total_class_images = class_image_counts(records, n_classes)
    total_positive = sum(record.positive for record in records)
    total_negative = total_images - total_positive
    total_sources = Counter(record.record.source_name for record in records)
    total_signatures = Counter(signature(record) for record in records)

    class_targets = {
        split: {
            class_id: total_class_images[class_id] * ratios[split]
            for class_id in range(n_classes)
        }
        for split in SPLITS
    }
    positive_targets = {
        split: total_positive * ratios[split] for split in SPLITS
    }
    negative_targets = {
        split: total_negative * ratios[split] for split in SPLITS
    }
    source_targets = {
        split: {
            source: total * ratios[split] for source, total in total_sources.items()
        }
        for split in SPLITS
    }
    signature_targets = {
        split: {
            sig: total * ratios[split] for sig, total in total_signatures.items()
        }
        for split in SPLITS
    }

    def group_rarity(group: list[TaskImage]) -> float:
        score = 0.0
        for record in group:
            for class_id in record.class_ids:
                score += 1.0 / max(total_class_images[class_id], 1)
            score += 0.2 / max(total_signatures[signature(record)], 1)
        return score

    rng.shuffle(groups)
    groups.sort(key=group_rarity, reverse=True)

    result = {split: [] for split in SPLITS}
    image_counts = Counter()
    class_counts: dict[str, Counter[int]] = {
        split: Counter() for split in SPLITS
    }
    positive_counts = Counter()
    negative_counts = Counter()
    source_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }
    signature_counts: dict[str, Counter[str]] = {
        split: Counter() for split in SPLITS
    }

    def squared_deviation(actual: float, target: float) -> float:
        scale = max(target, 1.0)
        return ((actual - target) / scale) ** 2

    for group in groups:
        group_size = len(group)
        group_class = Counter()
        group_positive = 0
        group_negative = 0
        group_sources = Counter()
        group_signatures = Counter()
        for record in group:
            for class_id in record.class_ids:
                group_class[class_id] += 1
            if record.positive:
                group_positive += 1
            else:
                group_negative += 1
            group_sources[record.record.source_name] += 1
            group_signatures[signature(record)] += 1

        feasible = [
            split
            for split in SPLITS
            if image_counts[split] + group_size <= image_targets[split]
        ]
        choices = feasible or list(SPLITS)
        best_split = None
        best_cost = float("inf")
        for split in choices:
            before = 30.0 * squared_deviation(
                image_counts[split], image_targets[split]
            )
            after = 30.0 * squared_deviation(
                image_counts[split] + group_size, image_targets[split]
            )
            delta = after - before

            for class_id, contribution in group_class.items():
                target = class_targets[split][class_id]
                delta += 25.0 * (
                    squared_deviation(
                        class_counts[split][class_id] + contribution, target
                    )
                    - squared_deviation(class_counts[split][class_id], target)
                )
            if group_positive:
                target = positive_targets[split]
                delta += 15.0 * (
                    squared_deviation(
                        positive_counts[split] + group_positive, target
                    )
                    - squared_deviation(positive_counts[split], target)
                )
            if group_negative:
                target = negative_targets[split]
                delta += 15.0 * (
                    squared_deviation(
                        negative_counts[split] + group_negative, target
                    )
                    - squared_deviation(negative_counts[split], target)
                )
            for source, contribution in group_sources.items():
                target = source_targets[split][source]
                delta += 2.0 * (
                    squared_deviation(
                        source_counts[split][source] + contribution, target
                    )
                    - squared_deviation(source_counts[split][source], target)
                )
            for sig, contribution in group_signatures.items():
                target = signature_targets[split][sig]
                delta += 3.0 * (
                    squared_deviation(
                        signature_counts[split][sig] + contribution, target
                    )
                    - squared_deviation(signature_counts[split][sig], target)
                )

            overflow = max(
                0, image_counts[split] + group_size - image_targets[split]
            )
            delta += 1_000_000.0 * overflow
            delta += rng.random() * 1e-7
            if delta < best_cost:
                best_cost = delta
                best_split = split

        assert best_split is not None
        result[best_split].extend(group)
        image_counts[best_split] += group_size
        class_counts[best_split].update(group_class)
        positive_counts[best_split] += group_positive
        negative_counts[best_split] += group_negative
        source_counts[best_split].update(group_sources)
        signature_counts[best_split].update(group_signatures)

    for split in SPLITS:
        rng.shuffle(result[split])
    return result

def split_score(
    candidate: dict[str, list[TaskImage]],
    ratios: dict[str, float],
    n_classes: int,
) -> tuple[float, dict[str, Any]]:
    all_records = [record for split in SPLITS for record in candidate[split]]
    total_images = len(all_records)
    total_class_images = class_image_counts(all_records, n_classes)
    total_instances = class_instance_counts(all_records, n_classes)
    total_positive = sum(record.positive for record in all_records)
    total_negative = total_images - total_positive
    total_sources = Counter(record.record.source_name for record in all_records)

    score = 0.0
    missing = 0

    def deviation(actual: int, total: int, target: float) -> float:
        return abs(actual / total - target) if total else 0.0

    for split in SPLITS:
        score += 20.0 * deviation(
            len(candidate[split]), total_images, ratios[split]
        )
        split_class_images = class_image_counts(candidate[split], n_classes)
        split_instances = class_instance_counts(candidate[split], n_classes)
        split_positive = sum(record.positive for record in candidate[split])
        split_negative = len(candidate[split]) - split_positive
        for class_id in range(n_classes):
            if total_class_images[class_id] >= 3 and split_class_images[class_id] == 0:
                missing += 1
            score += 15.0 * deviation(
                split_class_images[class_id],
                total_class_images[class_id],
                ratios[split],
            )
            score += 5.0 * deviation(
                split_instances[class_id], total_instances[class_id], ratios[split]
            )
        if total_positive >= 3 and split_positive == 0:
            missing += 1
        if total_negative >= 3 and split_negative == 0:
            missing += 1
        score += 10.0 * deviation(split_positive, total_positive, ratios[split])
        score += 10.0 * deviation(split_negative, total_negative, ratios[split])
        split_sources = Counter(
            record.record.source_name for record in candidate[split]
        )
        for source_name, total in total_sources.items():
            score += 1.0 * deviation(
                split_sources[source_name], total, ratios[split]
            )
    score += 1_000_000.0 * missing
    return score, {"missing_required_content": missing}


def optimize_split(
    records: list[TaskImage],
    ratios: dict[str, float],
    n_classes: int,
    seed: int,
    trials: int,
) -> tuple[dict[str, list[TaskImage]], dict[str, Any]]:
    if not records:
        raise DatasetError("No balanced records are available for splitting")
    best: dict[str, list[TaskImage]] | None = None
    best_score = float("inf")
    best_diagnostics: dict[str, Any] = {}
    seed_rng = random.Random(seed)
    for _ in range(max(1, trials)):
        rng = random.Random(seed_rng.randrange(0, 2**31 - 1))
        candidate = allocate_greedy(records, ratios, n_classes, rng)
        score, diagnostics = split_score(candidate, ratios, n_classes)
        if score < best_score:
            best = candidate
            best_score = score
            best_diagnostics = diagnostics
    assert best is not None
    return best, {
        "split_score": best_score,
        "split_trials": max(1, trials),
        **best_diagnostics,
    }


def bbox_line(class_id: int, annotation: Annotation) -> str:
    xc, yc, width, height = annotation.bbox_xywh
    return f"{class_id} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}"


def polygon_line(class_id: int, annotation: Annotation) -> str:
    if not annotation.polygon:
        raise DatasetError("Cannot write polygon line for bbox-only annotation")
    coordinates = " ".join(
        f"{value:.8f}" for point in annotation.polygon for value in point
    )
    return f"{class_id} {coordinates}"


def copy_or_link(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if mode == "hardlink":
        try:
            target.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def write_yaml(root: Path, names: Sequence[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    content = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def rasterize_instance_mask(
    target: TargetObject, width: int, height: int, output_path: Path
) -> None:
    if not target.annotation.polygon:
        return
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    points = [
        (
            min(width - 1, max(0, int(round(x * (width - 1))))),
            min(height - 1, max(0, int(round(y * (height - 1))))),
        )
        for x, y in target.annotation.polygon
    ]
    draw.polygon(points, fill=255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path)


def write_outputs(
    output_root: Path,
    split_records: dict[str, list[TaskImage]],
    output_names: Sequence[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    detection_root = output_root / "detection"
    pixel_root = output_root / "pixel_annotations"
    segmentation_root = output_root / "segmentation"
    write_yaml(detection_root, output_names)
    write_yaml(segmentation_root, output_names)

    manifest_rows: list[dict[str, Any]] = []
    for split, records in split_records.items():
        for record in records:
            output_name = record.record.output_name
            detection_image = detection_root / "images" / split / output_name
            detection_label = (
                detection_root / "labels" / split / Path(output_name).with_suffix(".txt")
            )
            copy_or_link(record.record.image_path, detection_image, args.copy_mode)
            detection_label.parent.mkdir(parents=True, exist_ok=True)
            detection_label.write_text(
                "\n".join(
                    bbox_line(target.output_class_id, target.annotation)
                    for target in record.targets
                )
                + ("\n" if record.targets else ""),
                encoding="utf-8",
            )

            # Preserve every true polygon, even when the image is not suitable for
            # complete segmentation training because it also contains bbox-only targets.
            polygon_targets = [
                target
                for target in record.targets
                if target.annotation.geometry == "polygon"
            ]
            polygon_label = (
                pixel_root / "labels" / split / Path(output_name).with_suffix(".txt")
            )
            polygon_label.parent.mkdir(parents=True, exist_ok=True)
            polygon_label.write_text(
                "\n".join(
                    polygon_line(target.output_class_id, target.annotation)
                    for target in polygon_targets
                )
                + ("\n" if polygon_targets else ""),
                encoding="utf-8",
            )
            if args.write_masks:
                for instance_index, target in enumerate(polygon_targets):
                    mask_name = (
                        f"{instance_index:04d}_"
                        f"{slugify(target.output_class_name)}.png"
                    )
                    rasterize_instance_mask(
                        target,
                        record.record.width,
                        record.record.height,
                        pixel_root
                        / "masks"
                        / split
                        / Path(output_name).stem
                        / mask_name,
                    )

            segmentation_eligible = not record.positive or record.all_targets_polygon
            if segmentation_eligible:
                segmentation_image = (
                    segmentation_root / "images" / split / output_name
                )
                segmentation_label = (
                    segmentation_root
                    / "labels"
                    / split
                    / Path(output_name).with_suffix(".txt")
                )
                copy_or_link(
                    record.record.image_path, segmentation_image, args.copy_mode
                )
                segmentation_label.parent.mkdir(parents=True, exist_ok=True)
                segmentation_label.write_text(
                    "\n".join(
                        polygon_line(target.output_class_id, target.annotation)
                        for target in record.targets
                    )
                    + ("\n" if record.targets else ""),
                    encoding="utf-8",
                )

            manifest_rows.append(
                {
                    "split": split,
                    "source": record.record.source_name,
                    "source_yaml": str(record.record.source_yaml),
                    "source_image": str(record.record.image_path),
                    "source_label": str(record.record.label_path),
                    "output_image": str(detection_image),
                    "sha256": record.record.sha256,
                    "context": record.record.context,
                    "positive": record.positive,
                    "target_class_ids": ";".join(
                        str(value) for value in sorted(record.class_ids)
                    ),
                    "target_class_names": ";".join(
                        output_names[value] for value in sorted(record.class_ids)
                    ),
                    "target_instances": len(record.targets),
                    "polygon_instances": sum(
                        target.annotation.geometry == "polygon"
                        for target in record.targets
                    ),
                    "bbox_instances": sum(
                        target.annotation.geometry == "bbox"
                        for target in record.targets
                    ),
                    "ignored_foreign_instances": len(
                        record.ignored_foreign_objects
                    ),
                    "segmentation_eligible": segmentation_eligible,
                    "width": record.record.width,
                    "height": record.record.height,
                }
            )
    return manifest_rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(
    split_records: dict[str, list[TaskImage]], output_names: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for split in (*SPLITS, "all"):
        records = (
            split_records[split]
            if split in split_records
            else [record for name in SPLITS for record in split_records[name]]
        )
        positives = sum(record.positive for record in records)
        summary_rows.append(
            {
                "split": split,
                "total_images": len(records),
                "positive_images": positives,
                "negative_images": len(records) - positives,
                "positive_fraction": positives / len(records) if records else 0.0,
                "negative_fraction": (
                    (len(records) - positives) / len(records) if records else 0.0
                ),
                "segmentation_eligible_images": sum(
                    (not record.positive) or record.all_targets_polygon
                    for record in records
                ),
            }
        )
        image_counts = class_image_counts(records, len(output_names))
        instance_counts = class_instance_counts(records, len(output_names))
        polygon_counts = Counter(
            target.output_class_id
            for record in records
            for target in record.targets
            if target.annotation.geometry == "polygon"
        )
        bbox_counts = Counter(
            target.output_class_id
            for record in records
            for target in record.targets
            if target.annotation.geometry == "bbox"
        )
        for class_id, class_name in enumerate(output_names):
            class_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "positive_images": image_counts[class_id],
                    "instances": instance_counts[class_id],
                    "polygon_instances": polygon_counts[class_id],
                    "bbox_instances": bbox_counts[class_id],
                }
            )
    return summary_rows, class_rows


def prepare_output(path: Path, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists():
        if not overwrite:
            raise DatasetError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Build {VARIANT_TITLE}: threshold classes by unique positive-image "
            "count, create a natural master dataset, convert "
            "polygons to bounding boxes, and preserve pixel annotations."
        )
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        nargs="+",
        required=True,
        help="One or more YOLO dataset YAML/YML files to merge.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=int,
        default=400,
        help="T: required unique positive-image count per original FO class. Default: 400.",
    )
    parser.add_argument(
        "--threshold-rule",
        choices=("at_least", "greater_than"),
        default="at_least",
        help="at_least means count >= T; greater_than means count > T.",
    )
    parser.add_argument(
        "--class-balance",
        choices=("undersample", "none"),
        default="none",
        help="Multilabel-aware positive-image balancing across output classes.",
    )
    parser.add_argument(
        "--class-target",
        type=int,
        default=None,
        help="Optional positive-image target per class; default=min class support.",
    )
    parser.add_argument(
        "--dataset-policy", choices=("natural", "balanced"), default="natural",
        help="natural keeps the complete master dataset; balanced is an optional legacy construction mode.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help="Requested negative images / selected positive images.",
    )
    parser.add_argument(
        "--negative-policy",
        choices=("clean", "task"),
        default="clean",
        help=(
            "T-Core only: clean uses images with no FO at all; task also allows "
            "rare-only images as hard negatives. T-Core+Other always uses clean."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=500)
    parser.add_argument(
        "--copy-mode", choices=("copy", "hardlink"), default="hardlink"
    )
    parser.add_argument(
        "--file-read-retries",
        type=int,
        default=6,
        help=(
            "Number of attempts for transient Windows image-read failures. "
            "Default: 6."
        ),
    )
    parser.add_argument(
        "--file-read-delay",
        type=float,
        default=0.5,
        help=(
            "Base delay in seconds between file-read retries; delay increases "
            "linearly with each attempt. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat missing labels as empty only when this is known to be correct.",
    )
    parser.add_argument(
        "--invalid-row-policy", choices=("error", "skip"), default="error"
    )
    parser.add_argument(
        "--keep-exact-duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep exact duplicates in the master dataset by default; use --no-keep-exact-duplicates to deduplicate.",
    )
    parser.add_argument(
        "--write-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write original-resolution binary mask per polygon instance.",
    )
    parser.add_argument(
        "--require-core-class", nargs="*", default=[],
        help="Class names that must satisfy T and remain independent core labels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.threshold < 0:
        parser.error("--threshold must be non-negative")
    if args.negative_ratio < 0:
        parser.error("--negative-ratio must be non-negative")
    if not (0 < args.train_ratio < 1):
        parser.error("--train-ratio must be between 0 and 1")
    if not (0 <= args.val_ratio < 1):
        parser.error("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1:
        parser.error("train-ratio + val-ratio must be less than 1")
    if args.split_trials <= 0:
        parser.error("--split-trials must be positive")
    if args.file_read_retries <= 0:
        parser.error("--file-read-retries must be positive")
    if args.file_read_delay < 0:
        parser.error("--file-read-delay must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    try:
        prepare_output(args.output, args.overwrite)
        reports_root = args.output / "reports"

        records, scan_warnings = scan_sources(args)
        records, duplicate_rows = remove_or_group_duplicates(
            records, deduplicate=not args.keep_exact_duplicates
        )
        LOGGER.info("Unique images after duplicate handling: %d", len(records))

        class_stats, _ = original_class_statistics(
            records, args.threshold, args.threshold_rule
        )
        if not class_stats:
            raise DatasetError(
                "No foreign-object classes were found after excluding Wagon/Container."
            )

        normalized_stats = {normalize_name(stats["class_name"]): stats for stats in class_stats.values()}
        for required_name in args.require_core_class:
            required_key = normalize_name(required_name)
            if required_key not in normalized_stats:
                raise DatasetError(f"Required core class is absent from the source inventory: {required_name}")
            if not normalized_stats[required_key]["frequent"]:
                support = normalized_stats[required_key]["positive_images"]
                raise DatasetError(
                    f"Required class {required_name!r} has {support} positive images and does not meet T={args.threshold}."
                )

        task_images, output_names, mapping, rare = construct_task(
            records, class_stats
        )
        LOGGER.info("Output classes: %s", ", ".join(output_names))

        if args.dataset_policy == "natural":
            selected = list(task_images)
            positives = [record for record in task_images if record.positive]
            negatives = [record for record in task_images if not record.positive]
            balance_info = {
                "dataset_policy": "natural_master",
                "available_positive_images": len(positives),
                "selected_positive_images": len(positives),
                "available_negative_images": len(negatives),
                "selected_negative_images": len(negatives),
                "achieved_negative_positive_ratio": (len(negatives) / len(positives) if positives else None),
                "note": "All valid images are retained. Training-only negative control is performed by the training scripts.",
            }
            excluded_rare_only = []
        else:
            selected, balance_info, excluded_rare_only = balance_dataset(
                task_images, len(output_names), args
            )
            balance_info["dataset_policy"] = "balanced_master_legacy"
        if not selected:
            raise DatasetError("Balancing selected zero images")

        ratios = {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        }
        split_records, split_info = optimize_split(
            selected,
            ratios,
            len(output_names),
            args.seed,
            args.split_trials,
        )

        manifest_rows = write_outputs(
            args.output, split_records, output_names, args
        )
        summary_rows, class_rows = distribution_rows(
            split_records, output_names
        )

        original_rows: list[dict[str, Any]] = []
        for normalized_name, stats in sorted(
            class_stats.items(), key=lambda item: item[1]["class_name"].casefold()
        ):
            if normalized_name in mapping:
                mapped_name = output_names[mapping[normalized_name]]
                mapped_id: int | str = mapping[normalized_name]
            else:
                mapped_name = "EXCLUDED"
                mapped_id = ""
            original_rows.append(
                {
                    **stats,
                    "mapped_output_id": mapped_id,
                    "mapped_output_name": mapped_name,
                }
            )

        write_csv(reports_root / "source_class_statistics.csv", original_rows)
        write_csv(reports_root / "dataset_distribution_by_split.csv", summary_rows)
        write_csv(reports_root / "class_distribution_by_split.csv", class_rows)
        write_csv(reports_root / "image_manifest.csv", manifest_rows)
        write_csv(
            reports_root / "duplicate_images.csv",
            duplicate_rows,
            fieldnames=("sha256", "kept_image", "duplicate_image", "removed"),
        )
        write_csv(
            reports_root / "rare_only_excluded_images.csv",
            [
                {
                    "source": record.record.source_name,
                    "image": str(record.record.image_path),
                    "ignored_classes": ";".join(
                        sorted(
                            {
                                obj.source_class_name
                                for obj in record.ignored_foreign_objects
                            }
                        )
                    ),
                }
                for record in excluded_rare_only
            ],
            fieldnames=("source", "image", "ignored_classes"),
        )
        (reports_root / "scan_warnings.txt").write_text(
            "\n".join(scan_warnings) + ("\n" if scan_warnings else ""),
            encoding="utf-8",
        )

        build_summary = {
            "builder_version": VERSION,
            "variant": VARIANT,
            "variant_title": VARIANT_TITLE,
            "threshold": args.threshold,
            "threshold_rule": args.threshold_rule,
            "threshold_basis": "unique_positive_images_per_original_class",
            "dataset_policy": args.dataset_policy,
            "required_core_classes": args.require_core_class,
            "input_yamls": [str(path.resolve()) for path in args.yaml],
            "input_images_after_duplicate_handling": len(records),
            "output_classes": {
                index: name for index, name in enumerate(output_names)
            },
            "frequent_original_classes": [
                stats["class_name"]
                for stats in class_stats.values()
                if stats["frequent"]
            ],
            "rare_original_classes": [
                class_stats[name]["class_name"] for name in sorted(rare)
            ],
            "balance": balance_info,
            "split": split_info,
            "ratios": ratios,
            "selected_images": len(selected),
            "warnings": len(scan_warnings),
            "output_layout": {
                "detection": "YOLO bbox dataset; source polygons converted to tight boxes",
                "pixel_annotations": "all preserved true polygons and per-instance masks",
                "segmentation": "clean YOLO segmentation subset: polygon-complete positives plus negatives",
            },
        }
        reports_root.mkdir(parents=True, exist_ok=True)
        (reports_root / "build_summary.json").write_text(
            json.dumps(build_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        LOGGER.info("Completed %s", VARIANT_TITLE)
        LOGGER.info("Output: %s", args.output.resolve())
        LOGGER.info(
            "Selected: %d positive + %d negative images",
            balance_info["selected_positive_images"],
            balance_info["selected_negative_images"],
        )
        for row in summary_rows:
            if row["split"] != "all":
                LOGGER.info(
                    "%s: %d images (%d positive, %d negative)",
                    row["split"],
                    row["total_images"],
                    row["positive_images"],
                    row["negative_images"],
                )
        return 0
    except DatasetError as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception:
        LOGGER.exception("Unexpected failure")
        return 3


if __name__ == "__main__":
    sys.exit(main())
