
from pathlib import Path
import shutil
import random
import csv
from collections import Counter, defaultdict

# ============================================================
# CREATE D1 CLASSIFICATION DATASET — CONTAINER vs WAGON
# CONTEXT IS DECIDED BY CLASS ID ONLY
# ============================================================
#
# IMPORTANT FIX:
# D1 is classification. Bounding boxes are not used for training.
# Therefore, an image is considered Container/Wagon if its label file contains
# class ID 1 or 22, even if the context bounding box coordinates are invalid.
#
# This reproduces the previous logic where only images without Container/Wagon
# labels are excluded.
#
# Expected with your diagnosis:
#   Total images: 3359
#   Included:     3334
#   Excluded:       25
#
# ============================================================

SOURCE_DATASET = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26")

OUTPUT_DATASET = Path(
    r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Focused_Datasets_With_Negatives_Optimized_80_10_10\D1_Context_Classification_Container_Wagon"
)

SOURCE_IMAGES_DIR = SOURCE_DATASET / "images"
SOURCE_LABELS_DIR = SOURCE_DATASET / "labels"

SUMMARY_CSV = OUTPUT_DATASET / "D1_Context_Classification_dataset_analysis.csv"

CONTAINER_ID = 1
WAGON_ID = 22

ORIGINAL_TO_NEW = {
    CONTAINER_ID: 0,
    WAGON_ID: 1,
}

CLASS_NAMES = {
    0: "Container",
    1: "Wagon",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

TARGET_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SPLITS = ["train", "val", "test"]

RANDOM_SEED = 42
N_TRIALS = 30000


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

            if len(parts) < 1:
                continue

            try:
                class_id = int(float(parts[0]))
                # We keep coordinates only for diagnostics. D1 does not use them.
                rows.append([class_id] + parts[1:])
            except ValueError:
                print(f"WARNING: invalid class ID in {label_path}, line {line_number}: {line}")
                continue

    return rows


def find_image_by_stem(stem: str):
    if not SOURCE_IMAGES_DIR.exists():
        return None

    for p in SOURCE_IMAGES_DIR.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p

    return None


def get_single_context_class_by_id(rows):
    """
    Returns:
      0 for Container
      1 for Wagon
      None for no context class ID
      "ambiguous" if both Container and Wagon class IDs are present

    No bbox validation is used here because D1 is classification.
    """
    context_classes = set()

    for row in rows:
        old_class_id = row[0]

        if old_class_id in ORIGINAL_TO_NEW:
            context_classes.add(ORIGINAL_TO_NEW[old_class_id])

    if len(context_classes) == 0:
        return None

    if len(context_classes) > 1:
        return "ambiguous"

    return next(iter(context_classes))


def reset_output_folder():
    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)

    for split in SPLITS:
        for class_name in CLASS_NAMES.values():
            (OUTPUT_DATASET / split / class_name).mkdir(parents=True, exist_ok=True)


def count_by_class(records):
    counts = Counter()
    for r in records:
        counts[r["class_id"]] += 1
    return counts


def make_candidate_split(records, rng):
    by_class = defaultdict(list)

    for r in records:
        by_class[r["class_id"]].append(r)

    split_records = {s: [] for s in SPLITS}

    for class_id, class_records in by_class.items():
        shuffled = list(class_records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = round(n * TARGET_RATIOS["train"])
        n_val = round(n * TARGET_RATIOS["val"])

        split_records["train"].extend(shuffled[:n_train])
        split_records["val"].extend(shuffled[n_train:n_train + n_val])
        split_records["test"].extend(shuffled[n_train + n_val:])

    for split in SPLITS:
        rng.shuffle(split_records[split])

    return split_records


def split_has_all_classes(split_records):
    for split in SPLITS:
        counts = count_by_class(split_records[split])
        for class_id in CLASS_NAMES:
            if counts[class_id] == 0:
                return False
    return True


def evaluate_split(split_records):
    all_records = []
    for split in SPLITS:
        all_records.extend(split_records[split])

    total_images = len(all_records)
    total_class_counts = count_by_class(all_records)

    score = 0.0

    for split in SPLITS:
        actual = len(split_records[split]) / total_images if total_images else 0
        score += abs(actual - TARGET_RATIOS[split]) * 2.0

    for class_id in CLASS_NAMES:
        total_class = total_class_counts[class_id]

        if total_class == 0:
            score += 100000
            continue

        for split in SPLITS:
            split_counts = count_by_class(split_records[split])
            actual = split_counts[class_id] / total_class
            score += abs(actual - TARGET_RATIOS[split]) * 10.0

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

        if split_has_all_classes(candidate) and score < best_valid_score:
            best_valid_score = score
            best_valid_split = candidate

    if best_valid_split is not None:
        return best_valid_split, best_valid_score, True

    return best_any_split, best_any_score, False


def collect_records():
    if not SOURCE_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {SOURCE_IMAGES_DIR}")

    if not SOURCE_LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels folder not found: {SOURCE_LABELS_DIR}")

    images = sorted([
        p for p in SOURCE_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    records = []

    missing_label = 0
    empty_label = 0
    no_context = 0
    ambiguous_context = 0
    raw_context_counts = Counter()

    for image_path in images:
        stem = image_path.stem
        label_path = SOURCE_LABELS_DIR / f"{stem}.txt"

        if not label_path.exists():
            missing_label += 1
            continue

        rows = read_yolo_label(label_path)

        if len(rows) == 0:
            empty_label += 1
            continue

        for row in rows:
            if row[0] in ORIGINAL_TO_NEW:
                raw_context_counts[row[0]] += 1

        class_id = get_single_context_class_by_id(rows)

        if class_id is None:
            no_context += 1
            continue

        if class_id == "ambiguous":
            ambiguous_context += 1
            continue

        records.append({
            "stem": stem,
            "image_path": image_path,
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
        })

    print("Filtering summary:")
    print(f"  Total image files:              {len(images)}")
    print(f"  Missing label files:            {missing_label}")
    print(f"  Empty label files:              {empty_label}")
    print(f"  Label exists but no context:    {no_context}")
    print(f"  Ambiguous Container+Wagon:      {ambiguous_context}")
    print(f"  Selected D1 images:             {len(records)}")
    print()
    print("Raw context label counts:")
    print(f"  Container original ID {CONTAINER_ID}: {raw_context_counts[CONTAINER_ID]}")
    print(f"  Wagon original ID {WAGON_ID}:         {raw_context_counts[WAGON_ID]}")
    print()

    counts = count_by_class(records)
    print("Selected image counts:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}: {counts[class_id]}")
    print()

    if len(records) == 0:
        raise RuntimeError("No valid D1 classification records found.")

    return records


def write_dataset(split_records):
    reset_output_folder()

    for split in SPLITS:
        for record in split_records[split]:
            dst_dir = OUTPUT_DATASET / split / record["class_name"]
            dst_path = dst_dir / record["image_path"].name
            shutil.copy2(record["image_path"], dst_path)


def analyze_and_write_csv(split_records):
    total_images = sum(len(split_records[s]) for s in SPLITS)
    rows = []

    print("=" * 100)
    print("FINAL D1 DATASET SUMMARY")
    print("=" * 100)
    print(f"Dataset path: {OUTPUT_DATASET}")
    print(f"Total images: {total_images}")
    print()

    for split in SPLITS:
        split_records_list = split_records[split]
        split_total = len(split_records_list)
        split_percent = split_total / total_images * 100 if total_images else 0

        print(f"{split}: {split_total} images ({split_percent:.2f}%)")

        for class_id, class_name in CLASS_NAMES.items():
            class_count = sum(1 for r in split_records_list if r["class_id"] == class_id)
            class_total = sum(
                1 for s in SPLITS for r in split_records[s]
                if r["class_id"] == class_id
            )
            class_percent = class_count / class_total * 100 if class_total else 0

            print(f"  {class_name}: {class_count} ({class_percent:.2f}% of {class_name})")

            rows.append({
                "split": split,
                "split_total_images": split_total,
                "split_percent": f"{split_percent:.3f}",
                "class_id": class_id,
                "class_name": class_name,
                "class_images": class_count,
                "class_percent": f"{class_percent:.3f}",
            })

        print()

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "split_total_images",
                "split_percent",
                "class_id",
                "class_name",
                "class_images",
                "class_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV summary written to:")
    print(f"  {SUMMARY_CSV}")
    print("=" * 100)
    print("RESULT: PASSED — D1 classification dataset is ready.")
    print("=" * 100)


def main():
    print("=" * 100)
    print("Creating D1 classification dataset: Container vs Wagon")
    print("Context is decided by class ID only, not bbox validity.")
    print("=" * 100)

    records = collect_records()

    print(f"Optimizing 80/10/10 stratified split with {N_TRIALS} trials...")
    split_records, score, valid = optimize_split(records)

    print(f"Best split score: {score:.6f}")
    if valid:
        print("Split validity: OK — train/val/test all contain both classes.")
    else:
        print("Split validity: WARNING — best effort split.")
    print()

    write_dataset(split_records)
    analyze_and_write_csv(split_records)


if __name__ == "__main__":
    main()
