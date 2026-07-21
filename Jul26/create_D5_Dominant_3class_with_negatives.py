
from pathlib import Path
import shutil
import random
import csv
import statistics
from collections import Counter, defaultdict

# ============================================================
# CREATE D5 DETECTION DATASET WITH NEGATIVE CONTEXT IMAGES
# CONTEXT IS DECIDED BY CLASS ID ONLY
# ============================================================
#
# IMPORTANT FIX:
# Context filtering uses only the presence of Container/Wagon class IDs.
# It does NOT require valid context bounding boxes.
#
# Reason:
# Context boxes are not targets in D2-D5. They are only used to select images
# from Container/Wagon scenes. Target boxes are still strictly validated.
#
# Empty label files are intentional for negative/background images.
#
# ============================================================

SOURCE_DATASET = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26")

OUTPUT_DATASET = Path(
    r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Focused_Datasets_With_Negatives_Optimized_80_10_10\D5_Dominant_3class_BAW_Plastiksack_Papier_with_negatives"
)

SOURCE_IMAGES_DIR = SOURCE_DATASET / "images"
SOURCE_LABELS_DIR = SOURCE_DATASET / "labels"

SUMMARY_CSV = OUTPUT_DATASET / "D5_Dominant_3class_with_negatives_analysis.csv"

CONTAINER_ID = 1
WAGON_ID = 22
CONTEXT_CLASS_IDS = {CONTAINER_ID, WAGON_ID}

CLASS_MAPPING = {9: 0, 6: 1, 4: 2}
CLASS_NAMES = {0: "BAW-Sack", 1: "Plastiksack", 2: "Papier"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

TARGET_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLITS = ["train", "val", "test"]

RANDOM_SEED = 42
N_TRIALS = 30000

NEGATIVE_RATIO_MODE = "all"
NEGATIVE_TO_POSITIVE_RATIO = 1.5


def is_valid_yolo_bbox(x, y, w, h):
    return 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1


def read_yolo_label(label_path: Path):
    rows = []

    if not label_path.exists():
        return rows

    with label_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                print(f"WARNING: malformed line in {label_path}, line {line_number}: {line}")
                continue

            try:
                class_id = int(float(parts[0]))
                x, y, w, h = [float(v) for v in parts[1:5]]
                rows.append([class_id, x, y, w, h])
            except ValueError:
                print(f"WARNING: invalid numeric value in {label_path}, line {line_number}: {line}")
                continue

    return rows


def find_image_by_stem(stem: str):
    if not SOURCE_IMAGES_DIR.exists():
        return None

    for p in SOURCE_IMAGES_DIR.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p

    return None


def find_images(images_dir: Path):
    if not images_dir.exists():
        return []

    return sorted([
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def has_context_by_class_id(rows):
    return any(row[0] in CONTEXT_CLASS_IDS for row in rows)


def extract_valid_target_rows(rows):
    new_rows = []
    removed_invalid = 0
    removed_invalid_by_original_class = Counter()

    for row in rows:
        old_class_id, x, y, w, h = row

        if old_class_id in CLASS_MAPPING:
            if is_valid_yolo_bbox(x, y, w, h):
                new_class_id = CLASS_MAPPING[old_class_id]
                new_rows.append([new_class_id, x, y, w, h])
            else:
                removed_invalid += 1
                removed_invalid_by_original_class[old_class_id] += 1

    return new_rows, removed_invalid, removed_invalid_by_original_class


def reset_output_folder():
    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)

    for split in SPLITS:
        (OUTPUT_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_label(label_path: Path, rows):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with label_path.open("w", encoding="utf-8") as f:
        for row in rows:
            class_id = int(row[0])
            x, y, w, h = row[1:5]
            f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def write_yaml():
    yaml_path = OUTPUT_DATASET / "data.yaml"

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(f"path: {OUTPUT_DATASET.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for class_id, class_name in CLASS_NAMES.items():
            f.write(f"  {class_id}: {class_name}\n")

    return yaml_path


def collect_records():
    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {SOURCE_IMAGES_DIR}")

    if not SOURCE_LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels folder not found: {SOURCE_LABELS_DIR}")

    images = sorted([
        p for p in SOURCE_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    positive_records = []
    negative_records = []

    missing_labels = 0
    empty_labels = 0
    excluded_no_context = 0
    context_images = 0
    removed_invalid_target_boxes = 0
    removed_invalid_by_original_class = Counter()
    original_target_counts = Counter()
    final_target_counts = Counter()

    for image_path in images:
        stem = image_path.stem
        label_path = SOURCE_LABELS_DIR / f"{stem}.txt"

        if not label_path.exists():
            missing_labels += 1
            continue

        rows = read_yolo_label(label_path)

        if len(rows) == 0:
            empty_labels += 1
            excluded_no_context += 1
            continue

        if not has_context_by_class_id(rows):
            excluded_no_context += 1
            continue

        context_images += 1

        for row in rows:
            if row[0] in CLASS_MAPPING:
                original_target_counts[row[0]] += 1

        new_rows, removed_invalid, removed_by_class = extract_valid_target_rows(rows)
        removed_invalid_target_boxes += removed_invalid
        removed_invalid_by_original_class.update(removed_by_class)

        for row in new_rows:
            final_target_counts[int(row[0])] += 1

        record = {
            "stem": stem,
            "image_path": image_path,
            "new_rows": new_rows,
        }

        if len(new_rows) > 0:
            positive_records.append(record)
        else:
            negative_records.append(record)

    if NEGATIVE_RATIO_MODE == "limited":
        max_neg = int(round(len(positive_records) * NEGATIVE_TO_POSITIVE_RATIO))
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(negative_records)
        negative_records = negative_records[:max_neg]

    records = positive_records + negative_records

    print("Filtering summary:")
    print(f"  Total image files:                 {len(images)}")
    print(f"  Missing label files:               {missing_labels}")
    print(f"  Empty label files:                 {empty_labels}")
    print(f"  Excluded no context ID:            {excluded_no_context}")
    print(f"  Context images retained:           {context_images}")
    print(f"  Positive images:                   {len(positive_records)}")
    print(f"  Negative images retained:          {len(negative_records)}")
    print(f"  Removed invalid target boxes:      {removed_invalid_target_boxes}")
    print()

    print("Original target counts before cleaning:")
    for original_id, new_id in CLASS_MAPPING.items():
        print(f"  Original ID {original_id} -> New ID {new_id} {CLASS_NAMES[new_id]}: {original_target_counts[original_id]}")
    print()

    print("Final target counts after cleaning/remapping:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  New ID {class_id} {class_name}: {final_target_counts[class_id]}")
    print()

    if removed_invalid_target_boxes > 0:
        print("Removed invalid target boxes by original class:")
        for original_id, count in sorted(removed_invalid_by_original_class.items()):
            print(f"  Original ID {original_id}: {count}")
        print()

    if len(records) == 0:
        raise RuntimeError("No records collected.")

    return records


def count_objects_by_class(records):
    counts = defaultdict(int)
    for record in records:
        for row in record["new_rows"]:
            counts[int(row[0])] += 1
    return {cid: counts[cid] for cid in CLASS_NAMES.keys()}


def count_positive_negative(records):
    pos = sum(1 for r in records if len(r["new_rows"]) > 0)
    neg = sum(1 for r in records if len(r["new_rows"]) == 0)
    return pos, neg


def make_candidate_split(records, rng):
    positives = [r for r in records if len(r["new_rows"]) > 0]
    negatives = [r for r in records if len(r["new_rows"]) == 0]

    rng.shuffle(positives)
    rng.shuffle(negatives)

    split_records = {s: [] for s in SPLITS}

    for group in [positives, negatives]:
        n = len(group)
        n_train = round(n * TARGET_RATIOS["train"])
        n_val = round(n * TARGET_RATIOS["val"])

        split_records["train"].extend(group[:n_train])
        split_records["val"].extend(group[n_train:n_train + n_val])
        split_records["test"].extend(group[n_train + n_val:])

    for split in SPLITS:
        rng.shuffle(split_records[split])

    return split_records


def split_has_all_required_content(split_records):
    for split in SPLITS:
        records = split_records[split]
        pos, neg = count_positive_negative(records)

        if pos == 0 or neg == 0:
            return False

        counts = count_objects_by_class(records)
        for class_id in CLASS_NAMES.keys():
            if counts[class_id] == 0:
                return False

    return True


def evaluate_split(split_records):
    all_records = []
    for split in SPLITS:
        all_records.extend(split_records[split])

    total_images = len(all_records)
    total_pos, total_neg = count_positive_negative(all_records)
    total_class_counts = count_objects_by_class(all_records)

    score = 0.0

    for split in SPLITS:
        actual = len(split_records[split]) / total_images if total_images else 0.0
        score += abs(actual - TARGET_RATIOS[split]) * 2.0

    for split in SPLITS:
        pos, neg = count_positive_negative(split_records[split])
        actual_pos = pos / total_pos if total_pos else 0.0
        actual_neg = neg / total_neg if total_neg else 0.0
        score += abs(actual_pos - TARGET_RATIOS[split]) * 8.0
        score += abs(actual_neg - TARGET_RATIOS[split]) * 8.0

    for class_id in CLASS_NAMES.keys():
        total_objects = total_class_counts[class_id]

        if total_objects == 0:
            score += 100000
            continue

        for split in SPLITS:
            split_counts = count_objects_by_class(split_records[split])
            actual_obj = split_counts[class_id] / total_objects
            score += abs(actual_obj - TARGET_RATIOS[split]) * 10.0

    return score


def optimize_split(records):
    best_valid_split = None
    best_valid_score = float("inf")
    best_any_split = None
    best_any_score = float("inf")

    base_rng = random.Random(RANDOM_SEED)

    for _ in range(N_TRIALS):
        trial_seed = base_rng.randint(0, 10**9)
        rng = random.Random(trial_seed)

        candidate = make_candidate_split(records, rng)
        score = evaluate_split(candidate)

        if score < best_any_score:
            best_any_score = score
            best_any_split = candidate

        if split_has_all_required_content(candidate) and score < best_valid_score:
            best_valid_score = score
            best_valid_split = candidate

    if best_valid_split is not None:
        return best_valid_split, best_valid_score, True

    return best_any_split, best_any_score, False


def write_dataset(split_records):
    reset_output_folder()

    for split in SPLITS:
        dst_images_dir = OUTPUT_DATASET / "images" / split
        dst_labels_dir = OUTPUT_DATASET / "labels" / split

        for record in split_records[split]:
            src_image = record["image_path"]
            dst_image = dst_images_dir / src_image.name
            dst_label = dst_labels_dir / f"{record['stem']}.txt"

            shutil.copy2(src_image, dst_image)
            write_label(dst_label, record["new_rows"])

    return write_yaml()


def summarize_numeric(values):
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    if len(values) == 1:
        return {"min": values[0], "max": values[0], "mean": values[0], "median": values[0], "std": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
    }


def find_images_in_dir(images_dir: Path):
    if not images_dir.exists():
        return []
    return sorted([
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def analyze_split(split):
    images_dir = OUTPUT_DATASET / "images" / split
    labels_dir = OUTPUT_DATASET / "labels" / split

    images = find_images_in_dir(images_dir)
    label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_files}

    objects_per_image = []
    class_counts = Counter()
    widths, heights, areas = [], [], []

    empty_label_files = 0
    invalid_class_objects = 0
    invalid_bbox_objects = 0

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        rows = read_yolo_label(label_path)

        if len(rows) == 0:
            empty_label_files += 1

        objects_per_image.append(len(rows))

        for row in rows:
            class_id, x, y, w, h = row
            class_counts[class_id] += 1

            if class_id not in CLASS_NAMES:
                invalid_class_objects += 1

            if not is_valid_yolo_bbox(x, y, w, h):
                invalid_bbox_objects += 1

            widths.append(w)
            heights.append(h)
            areas.append(w * h)

    total_images = len(images)
    total_objects = sum(objects_per_image)
    positive_images = sum(1 for n in objects_per_image if n > 0)
    negative_images = sum(1 for n in objects_per_image if n == 0)

    return {
        "split": split,
        "images": total_images,
        "label_files": len(label_files),
        "objects": total_objects,
        "positive_images": positive_images,
        "negative_images": negative_images,
        "objects_per_image_mean": total_objects / total_images if total_images else 0,
        "objects_per_positive_image_mean": total_objects / positive_images if positive_images else 0,
        "objects_per_image_min": min(objects_per_image) if objects_per_image else 0,
        "objects_per_image_max": max(objects_per_image) if objects_per_image else 0,
        "empty_label_files": empty_label_files,
        "missing_label_files": len(sorted(image_stems - label_stems)),
        "orphan_label_files": len(sorted(label_stems - image_stems)),
        "invalid_class_objects": invalid_class_objects,
        "invalid_bbox_objects": invalid_bbox_objects,
        "class_counts": dict(class_counts),
        "width_stats": summarize_numeric(widths),
        "height_stats": summarize_numeric(heights),
        "area_stats": summarize_numeric(areas),
    }


def write_csv(results):
    total_images = sum(r["images"] for r in results)
    total_objects = sum(r["objects"] for r in results)
    total_positive = sum(r["positive_images"] for r in results)
    total_negative = sum(r["negative_images"] for r in results)

    class_columns = [f"{CLASS_NAMES[c].replace('-', '_')}_objects" for c in CLASS_NAMES]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "split", "images", "image_percent",
            "positive_images", "positive_image_percent",
            "negative_images", "negative_image_percent",
            "label_files", "objects", "object_percent",
        ] + class_columns + [
            "objects_per_image_mean",
            "objects_per_positive_image_mean",
            "objects_per_image_min",
            "objects_per_image_max",
            "empty_label_files",
            "missing_label_files",
            "orphan_label_files",
            "invalid_class_objects",
            "invalid_bbox_objects",
            "bbox_width_mean",
            "bbox_height_mean",
            "bbox_area_mean",
            "bbox_area_median",
        ]
        writer.writerow(header)

        for r in results:
            row = [
                r["split"],
                r["images"],
                f"{(r['images'] / total_images * 100) if total_images else 0:.3f}",
                r["positive_images"],
                f"{(r['positive_images'] / total_positive * 100) if total_positive else 0:.3f}",
                r["negative_images"],
                f"{(r['negative_images'] / total_negative * 100) if total_negative else 0:.3f}",
                r["label_files"],
                r["objects"],
                f"{(r['objects'] / total_objects * 100) if total_objects else 0:.3f}",
            ]

            for class_id in CLASS_NAMES:
                row.append(r["class_counts"].get(class_id, 0))

            row += [
                f"{r['objects_per_image_mean']:.6f}",
                f"{r['objects_per_positive_image_mean']:.6f}",
                r["objects_per_image_min"],
                r["objects_per_image_max"],
                r["empty_label_files"],
                r["missing_label_files"],
                r["orphan_label_files"],
                r["invalid_class_objects"],
                r["invalid_bbox_objects"],
                f"{r['width_stats']['mean']:.6f}",
                f"{r['height_stats']['mean']:.6f}",
                f"{r['area_stats']['mean']:.6f}",
                f"{r['area_stats']['median']:.6f}",
            ]
            writer.writerow(row)


def print_total_summary(results, yaml_path):
    total_images = sum(r["images"] for r in results)
    total_labels = sum(r["label_files"] for r in results)
    total_objects = sum(r["objects"] for r in results)
    total_positive = sum(r["positive_images"] for r in results)
    total_negative = sum(r["negative_images"] for r in results)

    total_empty = sum(r["empty_label_files"] for r in results)
    total_missing = sum(r["missing_label_files"] for r in results)
    total_orphan = sum(r["orphan_label_files"] for r in results)
    total_invalid_class = sum(r["invalid_class_objects"] for r in results)
    total_invalid_bbox = sum(r["invalid_bbox_objects"] for r in results)

    total_class_counts = Counter()
    for r in results:
        total_class_counts.update(r["class_counts"])

    print("=" * 100)
    print("FINAL DATASET SUMMARY")
    print("=" * 100)
    print(f"Dataset path:        {OUTPUT_DATASET}")
    print(f"YAML path:           {yaml_path}")
    print(f"Total images:        {total_images}")
    print(f"Positive images:     {total_positive}")
    print(f"Negative images:     {total_negative}")
    print(f"Total label files:   {total_labels}")
    print(f"Total objects:       {total_objects}")
    print(f"Empty label files:   {total_empty}  <-- intentional negatives")
    print()

    print("Total class counts:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}: {total_class_counts[class_id]}")

    print()
    print("Split ratios:")
    for r in results:
        image_percent = r["images"] / total_images * 100 if total_images else 0
        object_percent = r["objects"] / total_objects * 100 if total_objects else 0
        positive_percent = r["positive_images"] / total_positive * 100 if total_positive else 0
        negative_percent = r["negative_images"] / total_negative * 100 if total_negative else 0
        class_part = ", ".join([f"{CLASS_NAMES[c]}={r['class_counts'].get(c, 0)}" for c in CLASS_NAMES])

        print(
            f"  {r['split']:5s}: "
            f"{r['images']:5d} images ({image_percent:6.2f}%), "
            f"pos={r['positive_images']:5d} ({positive_percent:6.2f}%), "
            f"neg={r['negative_images']:5d} ({negative_percent:6.2f}%), "
            f"objects={r['objects']:5d} ({object_percent:6.2f}%), "
            f"{class_part}"
        )

    print()
    print("Data integrity:")
    print(f"  Missing label files:   {total_missing}")
    print(f"  Orphan label files:    {total_orphan}")
    print(f"  Invalid class objects: {total_invalid_class}")
    print(f"  Invalid bbox objects:  {total_invalid_bbox}")

    all_splits_have_content = True
    for r in results:
        if r["positive_images"] == 0 or r["negative_images"] == 0:
            all_splits_have_content = False
        for class_id in CLASS_NAMES:
            if r["class_counts"].get(class_id, 0) == 0:
                all_splits_have_content = False

    passed = (
        total_missing == 0
        and total_orphan == 0
        and total_invalid_class == 0
        and total_invalid_bbox == 0
        and total_images == total_labels
        and total_images > 0
        and total_positive > 0
        and total_negative > 0
        and total_objects > 0
        and all_splits_have_content
    )

    print()
    print(f"CSV summary written to:")
    print(f"  {SUMMARY_CSV}")
    print()

    print("=" * 100)
    if passed:
        print("RESULT: PASSED — dataset includes valid positives and intentional negative empty labels.")
    else:
        print("RESULT: WARNING — dataset needs checking before training.")
    print("=" * 100)


def main():
    print("=" * 100)
    print("Creating D5: BAW-Sack + Plastiksack + Papier detector with negatives")
    print("Context is decided by class ID only. Target boxes are still validated.")
    print("=" * 100)

    records = collect_records()

    print(f"Optimizing 80/10/10 split with {N_TRIALS} trials...")
    split_records, score, valid = optimize_split(records)

    print(f"Best split score: {score:.6f}")
    if valid:
        print("Split validity: OK — train/val/test contain positives, negatives, and all target classes.")
    else:
        print("Split validity: WARNING — best effort split; check summary.")
    print()

    yaml_path = write_dataset(split_records)

    results = []
    for split in SPLITS:
        results.append(analyze_split(split))

    write_csv(results)
    print_total_summary(results, yaml_path)


if __name__ == "__main__":
    main()
