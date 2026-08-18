#!/usr/bin/env python3
r"""Verify and render classification or YOLO detection/segmentation datasets.

Designed for the generated dataset folders:
  D1: train/<class>, val/<class>, test/<class>
  D2-D5 / T-Core variants:
      detection/dataset.yaml
      detection/images/{train,val,test}
      detection/labels/{train,val,test}

The script can also be pointed directly at a detection or segmentation folder.
It renders all images by default and writes CSV/JSON verification reports.

Examples (PowerShell):
  python verify_render_dataset.py --dataset-root C:\path\to\D1 --overwrite
  python verify_render_dataset.py --dataset-root C:\path\to\D2 --overwrite
  python verify_render_dataset.py --dataset-root C:\path\to\D2\segmentation --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml pillow") from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required. Install it with: pip install pillow pyyaml") from exc

VERSION = "1.2.0"
LOGGER = logging.getLogger("dataset_verifier")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}
PALETTE = [
    (255, 72, 72),
    (55, 190, 105),
    (65, 135, 255),
    (255, 190, 60),
    (180, 90, 255),
    (40, 205, 205),
    (255, 100, 190),
    (150, 210, 65),
    (245, 125, 35),
    (120, 150, 230),
    (225, 65, 135),
    (90, 205, 145),
]


@dataclass
class Annotation:
    class_id: int
    kind: str  # bbox or polygon
    values: list[float]
    line_number: int


@dataclass
class ImageRecord:
    split: str
    image_path: Path
    label_path: Path | None
    relative_path: Path
    class_folder: str | None = None
    annotations: list[Annotation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rendered_path: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and render D1 classification or YOLO detection/segmentation datasets."
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset",
        type=Path,
        default=Path("."),
        help="Dataset folder. Default: current folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder. Default: <dataset-root>/verification_rendered",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "classification", "detection", "segmentation"),
        default="auto",
        help="Dataset type. 'auto' is recommended.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to verify, e.g. --splits train val test",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=0,
        help="Maximum rendered images per split; 0 means all images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed when a maximum is used.")
    parser.add_argument(
        "--line-width",
        type=int,
        default=0,
        help="Annotation line width. 0 selects an image-size-dependent width.",
    )
    parser.add_argument(
        "--draw-polygon-box",
        action="store_true",
        help="Also draw the tight bounding box around each polygon.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing verification output folder.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return exit code 2 when verification errors are found.",
    )
    return parser.parse_args()


def normalize_splits(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for raw in values:
        value = raw.lower().strip()
        if value == "all":
            return ["train", "val", "test"]
        for canonical, aliases in SPLIT_ALIASES.items():
            if value in aliases:
                if canonical not in normalized:
                    normalized.append(canonical)
                break
        else:
            raise ValueError(f"Unsupported split: {raw!r}")
    return normalized


def find_yaml(folder: Path) -> Path | None:
    preferred = ["dataset.yaml", "dataset.yml", "data.yaml", "data.yml"]
    for name in preferred:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    candidates = sorted(list(folder.glob("*.yaml")) + list(folder.glob("*.yml")))
    return candidates[0] if candidates else None


def has_classification_layout(folder: Path) -> bool:
    found_split = False
    for aliases in SPLIT_ALIASES.values():
        split_dir = next((folder / alias for alias in aliases if (folder / alias).is_dir()), None)
        if split_dir is None:
            continue
        found_split = True
        if any(path.is_dir() for path in split_dir.iterdir()):
            return True
    return found_split


def resolve_dataset_root(input_root: Path, requested_mode: str) -> tuple[Path, str]:
    root = input_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    if requested_mode == "classification":
        return root, "classification"
    if requested_mode in {"detection", "segmentation"}:
        if (root / requested_mode).is_dir():
            return (root / requested_mode), requested_mode
        return root, requested_mode

    # Auto mode: a task root normally contains detection/.
    detection_dir = root / "detection"
    if detection_dir.is_dir() and find_yaml(detection_dir) is not None:
        return detection_dir, "detection"

    # A direct detection/segmentation root has a YAML and images/labels folders.
    if find_yaml(root) is not None and (root / "images").is_dir() and (root / "labels").is_dir():
        inferred = "segmentation" if root.name.lower() == "segmentation" else "detection"
        return root, inferred

    if has_classification_layout(root):
        return root, "classification"

    raise ValueError(
        "Could not determine dataset type. Expected D1 class folders or a YOLO folder "
        "containing dataset.yaml, images/, and labels/."
    )


def load_yaml_names(yaml_path: Path) -> dict[int, str]:
    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    names = data.get("names")
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        result: dict[int, str] = {}
        for key, value in names.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid class ID in YAML names: {key!r}")
        return dict(sorted(result.items()))
    raise ValueError(f"YAML has no valid 'names' list/dictionary: {yaml_path}")


def locate_split_dir(base: Path, canonical_split: str) -> Path | None:
    for alias in SPLIT_ALIASES[canonical_split]:
        candidate = base / alias
        if candidate.is_dir():
            return candidate
    return None


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_float(token: str) -> float:
    value = float(token)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("non-finite number")
    return value


def parse_yolo_label(label_path: Path, valid_class_ids: set[int]) -> tuple[list[Annotation], list[str], list[str]]:
    annotations: list[Annotation] = []
    warnings: list[str] = []
    errors: list[str] = []

    if not label_path.exists():
        errors.append("missing_label")
        return annotations, warnings, errors
    if not label_path.is_file():
        errors.append("label_not_a_file")
        return annotations, warnings, errors

    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = label_path.read_text(encoding="latin-1")
            warnings.append("label_decoded_as_latin1")
        except Exception as exc:  # pragma: no cover
            errors.append(f"cannot_read_label:{exc}")
            return annotations, warnings, errors
    except Exception as exc:
        errors.append(f"cannot_read_label:{exc}")
        return annotations, warnings, errors

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            class_value = safe_float(tokens[0])
            class_id = int(class_value)
            if class_value != class_id:
                raise ValueError("class ID is not an integer")
            values = [safe_float(token) for token in tokens[1:]]
        except (ValueError, IndexError) as exc:
            errors.append(f"line_{line_number}:parse_error:{exc}")
            continue

        if class_id not in valid_class_ids:
            errors.append(f"line_{line_number}:unknown_class_id:{class_id}")

        if len(values) == 4:
            kind = "bbox"
            xc, yc, width, height = values
            if width <= 0 or height <= 0:
                errors.append(f"line_{line_number}:nonpositive_bbox_size")
            if any(value < 0 or value > 1 for value in values):
                errors.append(f"line_{line_number}:bbox_coordinate_out_of_range")
            annotations.append(Annotation(class_id, kind, values, line_number))
        elif len(values) >= 6 and len(values) % 2 == 0:
            kind = "polygon"
            if any(value < 0 or value > 1 for value in values):
                errors.append(f"line_{line_number}:polygon_coordinate_out_of_range")
            points = list(zip(values[0::2], values[1::2]))
            if len(set(points)) < 3:
                errors.append(f"line_{line_number}:polygon_has_fewer_than_3_unique_points")
            annotations.append(Annotation(class_id, kind, values, line_number))
        else:
            errors.append(
                f"line_{line_number}:invalid_token_count:{len(tokens)} "
                "(expected class+4 bbox values or class+>=6 polygon values)"
            )

    return annotations, warnings, errors


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont,
             foreground: tuple[int, int, int], background: tuple[int, int, int], padding: int = 3) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.rectangle((x, y, x + width + 2 * padding, y + height + 2 * padding), fill=background)
    draw.text((x + padding, y + padding), text, fill=foreground, font=font)


def class_color(class_id: int) -> tuple[int, int, int]:
    return PALETTE[class_id % len(PALETTE)]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def render_detection_record(
    record: ImageRecord,
    output_path: Path,
    class_names: dict[int, str],
    requested_line_width: int,
    draw_polygon_box: bool,
) -> None:
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    line_width = requested_line_width or max(2, round(min(width, height) / 300))
    font = get_font(max(12, round(min(width, height) / 45)))
    small_font = get_font(max(10, round(min(width, height) / 60)))

    for annotation in record.annotations:
        color = class_color(annotation.class_id)
        class_name = class_names.get(annotation.class_id, f"class_{annotation.class_id}")
        label = f"{annotation.class_id}: {class_name}"

        if annotation.kind == "bbox":
            xc, yc, box_w, box_h = annotation.values
            x1 = clamp((xc - box_w / 2) * width, 0, width - 1)
            y1 = clamp((yc - box_h / 2) * height, 0, height - 1)
            x2 = clamp((xc + box_w / 2) * width, 0, width - 1)
            y2 = clamp((yc + box_h / 2) * height, 0, height - 1)
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=line_width)
            text_y = max(0, int(y1) - 2 * (small_font.size if hasattr(small_font, "size") else 12))
            text_box(draw, (int(x1), text_y), label, small_font, (255, 255, 255), color)
        else:
            normalized_points = list(zip(annotation.values[0::2], annotation.values[1::2]))
            points = [
                (clamp(x * width, 0, width - 1), clamp(y * height, 0, height - 1))
                for x, y in normalized_points
            ]
            if len(points) >= 3:
                overlay_fill = color + (45,)
                draw.polygon(points, fill=overlay_fill)
                draw.line(points + [points[0]], fill=color + (255,), width=line_width, joint="curve")
                x_values = [point[0] for point in points]
                y_values = [point[1] for point in points]
                x1, y1, x2, y2 = min(x_values), min(y_values), max(x_values), max(y_values)
                if draw_polygon_box:
                    draw.rectangle((x1, y1, x2, y2), outline=color + (180,), width=max(1, line_width - 1))
                text_y = max(0, int(y1) - 2 * (small_font.size if hasattr(small_font, "size") else 12))
                text_box(draw, (int(x1), text_y), label + " [polygon]", small_font, (255, 255, 255), color)

    if not record.annotations and not record.errors:
        text_box(draw, (8, 8), "NEGATIVE — empty YOLO label", font, (255, 255, 255), (45, 45, 45))

    if record.errors:
        text_box(
            draw,
            (8, 8),
            f"VERIFY ERROR ({len(record.errors)})",
            font,
            (255, 255, 255),
            (200, 25, 25),
        )

    split_text = f"split={record.split} | objects={len(record.annotations)}"
    text_bbox = draw.textbbox((0, 0), split_text, font=small_font)
    banner_width = text_bbox[2] - text_bbox[0] + 8
    banner_height = text_bbox[3] - text_bbox[1] + 8
    draw.rectangle((width - banner_width, height - banner_height, width, height), fill=(0, 0, 0, 150))
    draw.text((width - banner_width + 4, height - banner_height + 4), split_text, fill=(255, 255, 255), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output_path)


def render_classification_record(record: ImageRecord, output_path: Path) -> None:
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    font = get_font(max(14, round(min(width, height) / 38)))
    class_name = record.class_folder or "UNKNOWN"
    digest = int(hashlib.sha1(class_name.encode("utf-8")).hexdigest()[:8], 16)
    color = PALETTE[digest % len(PALETTE)]
    text_box(draw, (8, 8), f"CLASS: {class_name}", font, (255, 255, 255), color)
    text_box(draw, (8, max(8, height - 40)), f"split={record.split}", get_font(12), (255, 255, 255), (30, 30, 30))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image, output_path)


def save_image(image: Image.Image, output_path: Path) -> None:
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=94, subsampling=0)
    elif suffix == ".png":
        image.save(output_path, optimize=True)
    else:
        # Preserve the file name but use PNG when Pillow cannot reliably write the input extension.
        png_path = output_path.with_suffix(".png")
        image.save(png_path, optimize=True)


def collect_classification_records(root: Path, splits: list[str]) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    global_errors: list[str] = []
    for split in splits:
        split_dir = locate_split_dir(root, split)
        if split_dir is None:
            global_errors.append(f"missing_split_directory:{split}")
            continue
        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if not class_dirs:
            global_errors.append(f"no_class_directories:{split_dir}")
            continue
        for class_dir in class_dirs:
            images = list_images(class_dir)
            if not images:
                global_errors.append(f"empty_class_directory:{class_dir}")
            for image_path in images:
                records.append(
                    ImageRecord(
                        split=split,
                        image_path=image_path,
                        label_path=None,
                        relative_path=image_path.relative_to(split_dir),
                        class_folder=class_dir.name,
                    )
                )
    return records, global_errors


def collect_detection_records(
    root: Path,
    splits: list[str],
    class_names: dict[int, str],
) -> tuple[list[ImageRecord], list[str], list[Path]]:
    records: list[ImageRecord] = []
    global_errors: list[str] = []
    orphan_labels: list[Path] = []
    valid_ids = set(class_names)

    images_base = root / "images"
    labels_base = root / "labels"
    if not images_base.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_base}")
    if not labels_base.is_dir():
        raise FileNotFoundError(f"Missing labels folder: {labels_base}")

    for split in splits:
        image_split = locate_split_dir(images_base, split)
        label_split = locate_split_dir(labels_base, split)
        if image_split is None:
            global_errors.append(f"missing_image_split:{split}")
            continue
        if label_split is None:
            global_errors.append(f"missing_label_split:{split}")
            continue

        image_paths = list_images(image_split)
        image_stems: set[Path] = set()
        for image_path in image_paths:
            relative = image_path.relative_to(image_split)
            # Replace only the real image extension. Do not call with_suffix()
            # twice because filenames may contain meaningful dots in their stem,
            # e.g. Q3052-7.1c.jpg -> Q3052-7.1c.txt.
            relative_stem = relative.with_suffix("")
            image_stems.add(relative_stem)
            label_path = (label_split / relative).with_suffix(".txt")
            annotations, warnings, errors = parse_yolo_label(label_path, valid_ids)
            records.append(
                ImageRecord(
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    relative_path=relative,
                    annotations=annotations,
                    warnings=warnings,
                    errors=errors,
                )
            )

        for label_path in sorted(label_split.rglob("*.txt")):
            relative_label_stem = label_path.relative_to(label_split).with_suffix("")
            if relative_label_stem not in image_stems:
                orphan_labels.append(label_path)

    return records, global_errors, orphan_labels


def sample_records(records: list[ImageRecord], max_per_split: int, seed: int) -> list[ImageRecord]:
    if max_per_split <= 0:
        return records
    rng = random.Random(seed)
    by_split: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)
    selected: list[ImageRecord] = []
    for split, split_records in by_split.items():
        if len(split_records) <= max_per_split:
            selected.extend(split_records)
        else:
            selected.extend(rng.sample(split_records, max_per_split))
    return sorted(selected, key=lambda item: (item.split, str(item.relative_path).lower()))


def write_reports(
    output_root: Path,
    mode: str,
    dataset_root: Path,
    records_all: list[ImageRecord],
    records_rendered: list[ImageRecord],
    class_names: dict[int, str],
    global_errors: list[str],
    orphan_labels: list[Path],
) -> dict[str, Any]:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    rendered_keys = {(record.split, str(record.relative_path)) for record in records_rendered}
    image_csv = reports / "verification_images.csv"
    with image_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "split",
                "relative_image",
                "image_path",
                "label_path",
                "classification_class",
                "annotation_count",
                "bbox_count",
                "polygon_count",
                "class_instance_counts",
                "status",
                "warnings",
                "errors",
                "rendered",
                "rendered_path",
            ]
        )
        for record in records_all:
            counts = Counter(annotation.class_id for annotation in record.annotations)
            status = "ERROR" if record.errors else ("NEGATIVE" if mode != "classification" and not record.annotations else "OK")
            writer.writerow(
                [
                    record.split,
                    str(record.relative_path),
                    str(record.image_path),
                    str(record.label_path) if record.label_path else "",
                    record.class_folder or "",
                    len(record.annotations),
                    sum(annotation.kind == "bbox" for annotation in record.annotations),
                    sum(annotation.kind == "polygon" for annotation in record.annotations),
                    json.dumps({str(key): value for key, value in sorted(counts.items())}),
                    status,
                    " | ".join(record.warnings),
                    " | ".join(record.errors),
                    (record.split, str(record.relative_path)) in rendered_keys,
                    str(record.rendered_path) if record.rendered_path else "",
                ]
            )

    class_csv = reports / "class_distribution_by_split.csv"
    image_class_presence: Counter[tuple[str, str]] = Counter()
    instance_counts: Counter[tuple[str, str]] = Counter()
    negative_counts: Counter[str] = Counter()
    total_images: Counter[str] = Counter()

    for record in records_all:
        total_images[record.split] += 1
        if mode == "classification":
            class_name = record.class_folder or "UNKNOWN"
            image_class_presence[(record.split, class_name)] += 1
            instance_counts[(record.split, class_name)] += 1
        else:
            if not record.annotations and not record.errors:
                negative_counts[record.split] += 1
            present_ids = {annotation.class_id for annotation in record.annotations}
            for class_id in present_ids:
                image_class_presence[(record.split, class_names.get(class_id, f"class_{class_id}"))] += 1
            for annotation in record.annotations:
                instance_counts[(record.split, class_names.get(annotation.class_id, f"class_{annotation.class_id}"))] += 1

    class_labels = sorted({name for _, name in image_class_presence} | {name for _, name in instance_counts})
    with class_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "class", "positive_images", "instances"])
        for split in ("train", "val", "test"):
            for class_name in class_labels:
                writer.writerow(
                    [
                        split,
                        class_name,
                        image_class_presence[(split, class_name)],
                        instance_counts[(split, class_name)],
                    ]
                )

    split_csv = reports / "dataset_distribution_by_split.csv"
    with split_csv.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "total_images", "positive_images", "negative_images", "images_with_errors"])
        for split in ("train", "val", "test"):
            split_records = [record for record in records_all if record.split == split]
            errors = sum(bool(record.errors) for record in split_records)
            negatives = 0 if mode == "classification" else negative_counts[split]
            positives = total_images[split] - negatives - errors
            writer.writerow([split, total_images[split], positives, negatives, errors])

    error_lines = list(global_errors)
    error_lines.extend(f"orphan_label:{path}" for path in orphan_labels)
    for record in records_all:
        for warning in record.warnings:
            error_lines.append(f"WARNING:{record.image_path}:{warning}")
        for error in record.errors:
            error_lines.append(f"ERROR:{record.image_path}:{error}")
    (reports / "verification_errors.txt").write_text(
        "\n".join(error_lines) + ("\n" if error_lines else ""), encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "mode": mode,
        "class_names": class_names,
        "total_images": len(records_all),
        "rendered_images": len(records_rendered),
        "image_errors": sum(bool(record.errors) for record in records_all),
        "warnings": sum(len(record.warnings) for record in records_all),
        "global_errors": len(global_errors),
        "orphan_labels": len(orphan_labels),
        "empty_label_negatives": (
            0
            if mode == "classification"
            else sum(1 for record in records_all if not record.annotations and not record.errors)
        ),
        "annotations": sum(len(record.annotations) for record in records_all),
        "bbox_annotations": sum(
            annotation.kind == "bbox" for record in records_all for annotation in record.annotations
        ),
        "polygon_annotations": sum(
            annotation.kind == "polygon" for record in records_all for annotation in record.annotations
        ),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_records = [record for record in records_all if record.split == split]
        summary["splits"][split] = {
            "images": len(split_records),
            "rendered": sum(record in records_rendered for record in split_records),
            "errors": sum(bool(record.errors) for record in split_records),
            "negatives": (
                0
                if mode == "classification"
                else sum(not record.annotations and not record.errors for record in split_records)
            ),
            "annotations": sum(len(record.annotations) for record in split_records),
        }

    (reports / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        splits = normalize_splits(args.splits)
        input_root = args.dataset_root.expanduser().resolve()
        dataset_root, mode = resolve_dataset_root(input_root, args.mode)

        output_root = args.output.expanduser().resolve() if args.output else input_root / "verification_rendered"
        if output_root.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {output_root}. Use --overwrite to replace it."
                )
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Input root: %s", input_root)
        LOGGER.info("Resolved dataset root: %s", dataset_root)
        LOGGER.info("Detected mode: %s", mode)
        LOGGER.info("Output: %s", output_root)

        class_names: dict[int, str] = {}
        orphan_labels: list[Path] = []
        if mode == "classification":
            records_all, global_errors = collect_classification_records(dataset_root, splits)
        else:
            yaml_path = find_yaml(dataset_root)
            if yaml_path is None:
                raise FileNotFoundError(f"No dataset YAML/YML found in {dataset_root}")
            class_names = load_yaml_names(yaml_path)
            LOGGER.info("Class names: %s", class_names)
            records_all, global_errors, orphan_labels = collect_detection_records(
                dataset_root, splits, class_names
            )

        records_rendered = sample_records(records_all, args.max_images_per_split, args.seed)
        rendered_key_set = {(record.split, str(record.relative_path)) for record in records_rendered}

        for index, record in enumerate(records_all, start=1):
            should_render = (record.split, str(record.relative_path)) in rendered_key_set
            if not should_render:
                continue
            output_path = output_root / "rendered" / record.split / record.relative_path
            record.rendered_path = output_path
            try:
                if mode == "classification":
                    render_classification_record(record, output_path)
                else:
                    render_detection_record(
                        record,
                        output_path,
                        class_names,
                        args.line_width,
                        args.draw_polygon_box,
                    )
            except Exception as exc:
                record.errors.append(f"render_failure:{exc}")
                LOGGER.error("Could not render %s: %s", record.image_path, exc)
            if index % 250 == 0:
                LOGGER.info("Processed %d/%d images", index, len(records_all))

        summary = write_reports(
            output_root,
            mode,
            dataset_root,
            records_all,
            records_rendered,
            class_names,
            global_errors,
            orphan_labels,
        )

        LOGGER.info("Completed: %d images, %d rendered", summary["total_images"], summary["rendered_images"])
        LOGGER.info(
            "Annotations: %d bbox, %d polygon; negatives: %d",
            summary["bbox_annotations"],
            summary["polygon_annotations"],
            summary["empty_label_negatives"],
        )
        LOGGER.info(
            "Errors: %d image errors, %d global errors, %d orphan labels",
            summary["image_errors"],
            summary["global_errors"],
            summary["orphan_labels"],
        )
        LOGGER.info("Reports: %s", output_root / "reports")

        total_errors = summary["image_errors"] + summary["global_errors"] + summary["orphan_labels"]
        if args.fail_on_error and total_errors:
            return 2
        return 0

    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
