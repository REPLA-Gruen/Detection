
from pathlib import Path
from collections import Counter, defaultdict
import csv

# ============================================================
# LIGHT D1 DIAGNOSIS — NO HUGE FILES
# ============================================================
# This script does NOT write a per-image mega CSV.
# It writes only:
#   1. a short summary CSV
#   2. small TXT lists of stems by reason
#   3. optional CSV only for excluded images, without raw label lines
# ============================================================

SOURCE_DATASET = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26")

IMAGES_DIR = SOURCE_DATASET / "images"
LABELS_DIR = SOURCE_DATASET / "labels"

OUTPUT_DIR = SOURCE_DATASET / "_D1_diagnosis_LIGHT"

CONTAINER_ID = 1
WAGON_ID = 22

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Set this False if even the excluded-images CSV is too much.
WRITE_EXCLUDED_CSV = True


def is_valid_yolo_bbox(x, y, w, h):
    return 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1


def read_label_counts(label_path: Path):
    """
    Reads only counts. Does not store raw label lines.
    """
    counts = {
        "valid_container": 0,
        "valid_wagon": 0,
        "invalid_container": 0,
        "invalid_wagon": 0,
        "other": 0,
        "malformed": 0,
        "non_numeric": 0,
        "total_rows": 0,
    }

    if not label_path.exists():
        return counts

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()

            if not raw:
                continue

            parts = raw.split()

            if len(parts) < 5:
                counts["malformed"] += 1
                continue

            try:
                cls = int(float(parts[0]))
                x, y, w, h = [float(v) for v in parts[1:5]]
            except ValueError:
                counts["non_numeric"] += 1
                continue

            counts["total_rows"] += 1

            if cls == CONTAINER_ID:
                if is_valid_yolo_bbox(x, y, w, h):
                    counts["valid_container"] += 1
                else:
                    counts["invalid_container"] += 1
            elif cls == WAGON_ID:
                if is_valid_yolo_bbox(x, y, w, h):
                    counts["valid_wagon"] += 1
                else:
                    counts["invalid_wagon"] += 1
            else:
                counts["other"] += 1

    return counts


def get_reason_for_image(image_path: Path):
    stem = image_path.stem
    label_path = LABELS_DIR / f"{stem}.txt"

    if not label_path.exists():
        return "NO_LABEL_FILE", ""

    c = read_label_counts(label_path)

    has_container = c["valid_container"] > 0
    has_wagon = c["valid_wagon"] > 0

    if has_container and has_wagon:
        return "AMBIGUOUS_BOTH_CONTAINER_AND_WAGON", ""

    if has_container:
        return "INCLUDED_CONTAINER", "Container"

    if has_wagon:
        return "INCLUDED_WAGON", "Wagon"

    if c["invalid_container"] > 0 or c["invalid_wagon"] > 0:
        return "ONLY_INVALID_CONTEXT_BOXES", ""

    if c["total_rows"] == 0:
        return "EMPTY_LABEL_FILE", ""

    return "LABEL_EXISTS_BUT_NO_CONTEXT_CLASS", ""


def write_list(path: Path, stems):
    with path.open("w", encoding="utf-8") as f:
        for stem in sorted(stems):
            f.write(stem + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    labels = sorted(LABELS_DIR.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    reason_counter = Counter()
    class_counter = Counter()
    stems_by_reason = defaultdict(list)

    excluded_rows = []

    for image_path in images:
        reason, d1_class = get_reason_for_image(image_path)
        stem = image_path.stem

        reason_counter[reason] += 1
        stems_by_reason[reason].append(stem)

        if reason.startswith("INCLUDED_"):
            class_counter[d1_class] += 1
        else:
            excluded_rows.append({
                "stem": stem,
                "reason": reason,
                "has_label": int((LABELS_DIR / f"{stem}.txt").exists()),
            })

    included_total = reason_counter["INCLUDED_CONTAINER"] + reason_counter["INCLUDED_WAGON"]
    excluded_total = len(images) - included_total

    print("=" * 100)
    print("LIGHT D1 DIAGNOSIS")
    print("=" * 100)
    print(f"Source dataset: {SOURCE_DATASET}")
    print()
    print("Raw file counts:")
    print(f"  Total image files:             {len(images)}")
    print(f"  Total label files:             {len(labels)}")
    print(f"  Images with matching labels:    {len(image_stems & label_stems)}")
    print(f"  Images without label files:     {len(image_stems - label_stems)}")
    print(f"  Label files without images:     {len(label_stems - image_stems)}")
    print()
    print("D1 result:")
    print(f"  Included images:                {included_total}")
    print(f"  Excluded images:                {excluded_total}")
    print(f"  Container images:               {class_counter['Container']}")
    print(f"  Wagon images:                   {class_counter['Wagon']}")
    print()
    print("Reasons:")
    for reason, count in reason_counter.most_common():
        print(f"  {reason}: {count}")

    # Summary CSV
    summary_csv = OUTPUT_DIR / "D1_LIGHT_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "count"])
        writer.writerow(["total_images", len(images)])
        writer.writerow(["total_labels", len(labels)])
        writer.writerow(["images_with_matching_labels", len(image_stems & label_stems)])
        writer.writerow(["images_without_label_files", len(image_stems - label_stems)])
        writer.writerow(["label_files_without_images", len(label_stems - image_stems)])
        writer.writerow(["included_total", included_total])
        writer.writerow(["excluded_total", excluded_total])
        writer.writerow(["included_container", class_counter["Container"]])
        writer.writerow(["included_wagon", class_counter["Wagon"]])
        for reason, count in reason_counter.most_common():
            writer.writerow([reason, count])

    # Stem lists by reason
    for reason, stems in stems_by_reason.items():
        safe = reason.lower().replace(" ", "_").replace("/", "_")
        write_list(OUTPUT_DIR / f"{safe}.txt", stems)

    write_list(OUTPUT_DIR / "image_stems_without_label.txt", image_stems - label_stems)
    write_list(OUTPUT_DIR / "label_stems_without_image.txt", label_stems - image_stems)

    if WRITE_EXCLUDED_CSV:
        excluded_csv = OUTPUT_DIR / "D1_excluded_only.csv"
        with excluded_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "reason", "has_label"])
            writer.writeheader()
            writer.writerows(excluded_rows)

    print()
    print("Written outputs:")
    print(f"  {summary_csv}")
    print(f"  {OUTPUT_DIR}")
    print()
    print("This version should be very small, not hundreds of MB.")
    print("=" * 100)


if __name__ == "__main__":
    main()
