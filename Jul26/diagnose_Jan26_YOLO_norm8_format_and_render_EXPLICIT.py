
from pathlib import Path
import csv
import json
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

# ============================================================
# JAN-26 YOLO_norm8 DIAGNOSIS + RENDERING
# EXPLICIT PATH VERSION
# ============================================================
#
# This version is based exactly on your shown folder structure:
#
# C:\Hadi\Foreign_Material\Data\Ours\Jan-26\YOLO_norm8
# ├── images
# │   ├── train
# │   ├── val
# │   └── test
# ├── labels
# │   ├── train
# │   ├── val
# │   └── test
# └── dataset_norm8.yaml
#
# It renders all ground-truth labels and creates crops by class.
#
# ============================================================

YOLO_ROOT = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jan-26\YOLO_norm8")
DATASET_YAML = YOLO_ROOT / "dataset_norm8.yaml"

IMAGES_ROOT = YOLO_ROOT / "images"
LABELS_ROOT = YOLO_ROOT / "labels"

OUTPUT_DIR = YOLO_ROOT / "_diagnosis_render_norm8"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

SPLITS = ["train", "val", "test"]

SAVE_OBJECT_CROPS = True
MAX_CROPS_PER_CLASS = 500

FONT_SCALE = 0.75
FONT_THICK = 2

EXPECTED_NORM8_NAMES = {
    0: "BAW-Sack",
    1: "Plastic",
    2: "Metal",
    3: "Glass",
    4: "Paper_Cardboard",
    5: "Other",
    6: "Wagon",
    7: "Container",
}

# For comparison only — this is the raw Jun-26 schema.
RAW23_REFERENCE_NAMES = {
    0: "Schnur",
    1: "Container",
    2: "Zigaretten",
    3: "Kaffeekapsel",
    4: "Papier",
    5: "Katzensand",
    6: "Plastiksack",
    7: "Lebensmittelverpackung",
    8: "Alubuechse",
    9: "BAW-Sack",
    10: "Glasflasche",
    11: "Karton",
    12: "Alufolie",
    13: "PET 0.5l",
    14: "Textilien",
    15: "Metalle",
    16: "Windel",
    17: "PET 1.5l",
    18: "Styropor/Dämmung",
    19: "Blumentopf",
    20: "Diverses",
    21: "Gummiband",
    22: "Wagon",
}


def load_names_from_yaml():
    if not DATASET_YAML.exists():
        print(f"[WARN] YAML not found, using expected norm8 names: {DATASET_YAML}")
        return dict(EXPECTED_NORM8_NAMES)

    if yaml is not None:
        with DATASET_YAML.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        names = cfg.get("names", EXPECTED_NORM8_NAMES)

        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}

        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}

    # fallback minimal parser
    names = {}
    in_names = False
    for raw in DATASET_YAML.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue

        if s.startswith("names:"):
            in_names = True
            continue

        if in_names:
            if not raw.startswith(" ") and not raw.startswith("\t"):
                break
            if ":" in s:
                k, v = s.split(":", 1)
                try:
                    names[int(k.strip())] = v.strip().strip("'\"")
                except ValueError:
                    pass

    return names if names else dict(EXPECTED_NORM8_NAMES)


def color_for_class(cid):
    rng = random.Random(cid * 1013 + 19)
    return (
        rng.randint(40, 255),
        rng.randint(40, 255),
        rng.randint(40, 255),
    )


def list_images(split):
    img_dir = IMAGES_ROOT / split

    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    return sorted([
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def read_label_file(label_path):
    rows = []

    if not label_path.exists():
        return rows

    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()

            if not raw:
                continue

            parts = raw.split()

            if len(parts) < 5:
                rows.append({
                    "ok": False,
                    "error": "less_than_5_columns",
                    "line_no": line_no,
                    "raw": raw,
                })
                continue

            try:
                cid = int(float(parts[0]))
                x, y, w, h = [float(v) for v in parts[1:5]]
            except ValueError:
                rows.append({
                    "ok": False,
                    "error": "numeric_error",
                    "line_no": line_no,
                    "raw": raw,
                })
                continue

            valid_bbox = 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            valid_class_norm8 = 0 <= cid <= 7
            valid_class_raw23 = 0 <= cid <= 22

            rows.append({
                "ok": True,
                "error": "",
                "line_no": line_no,
                "raw": raw,
                "class_id": cid,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "valid_bbox_yolo_norm": valid_bbox,
                "valid_class_norm8": valid_class_norm8,
                "valid_class_raw23": valid_class_raw23,
                "n_cols": len(parts),
            })

    return rows


def clip_xyxy(box, W, H):
    x1, y1, x2, y2 = box

    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(1, min(W, int(round(x2))))
    y2 = max(1, min(H, int(round(y2))))

    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)

    return x1, y1, x2, y2


def yolo_norm_center_to_xyxy(row, W, H):
    x, y, w, h = row["x"], row["y"], row["w"], row["h"]

    return clip_xyxy(
        (
            (x - w / 2) * W,
            (y - h / 2) * H,
            (x + w / 2) * W,
            (y + h / 2) * H,
        ),
        W,
        H,
    )


def draw_label(img, text, x, y, color_bgr):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, FONT_SCALE, FONT_THICK)

    y = max(th + 6, y)

    cv2.rectangle(img, (x, y - th - 6), (x + tw + 8, y + 4), color_bgr, -1)
    cv2.putText(img, text, (x + 4, y - 2), font, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)


def render_image(img_path, rows, class_names, out_path):
    img = cv2.imread(str(img_path))

    if img is None:
        return False

    H, W = img.shape[:2]

    for r in rows:
        if not r.get("ok") or not r.get("valid_bbox_yolo_norm"):
            continue

        cid = r["class_id"]
        cname = class_names.get(cid, f"class_{cid}")

        col_rgb = color_for_class(cid)
        col_bgr = (col_rgb[2], col_rgb[1], col_rgb[0])

        x1, y1, x2, y2 = yolo_norm_center_to_xyxy(r, W, H)

        thick = max(2, int(round(min(W, H) / 230)))
        cv2.rectangle(img, (x1, y1), (x2, y2), col_bgr, thick)

        label = f"{cid}:{cname}"
        draw_label(img, label, x1, max(y1, 20), col_bgr)

    cv2.rectangle(img, (0, 0), (min(W, 760), 32), (255, 255, 255), -1)
    cv2.putText(
        img,
        f"Jan-26 YOLO_norm8 GT render: {img_path.name}",
        (5, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

    return True


def save_object_crops(img_path, rows, class_names, split, crop_counts):
    if not SAVE_OBJECT_CROPS:
        return 0

    img = cv2.imread(str(img_path))

    if img is None:
        return 0

    H, W = img.shape[:2]
    saved = 0

    for r in rows:
        if not r.get("ok") or not r.get("valid_bbox_yolo_norm"):
            continue

        cid = r["class_id"]

        if MAX_CROPS_PER_CLASS is not None and crop_counts[cid] >= MAX_CROPS_PER_CLASS:
            continue

        x1, y1, x2, y2 = yolo_norm_center_to_xyxy(r, W, H)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        cname = class_names.get(cid, f"class_{cid}")
        safe_name = cname.replace("/", "_").replace("\\", "_").replace(" ", "_")

        out_dir = OUTPUT_DIR / "crops_by_class" / f"{cid:02d}_{safe_name}" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{img_path.stem}_line{r['line_no']:03d}.jpg"
        cv2.imwrite(str(out_path), crop)

        crop_counts[cid] += 1
        saved += 1

    return saved


def main():
    print("=" * 100)
    print("Jan-26 YOLO_norm8 diagnosis and rendering")
    print("=" * 100)
    print(f"YOLO root: {YOLO_ROOT}")
    print(f"Images:    {IMAGES_ROOT}")
    print(f"Labels:    {LABELS_ROOT}")
    print(f"YAML:      {DATASET_YAML}")
    print(f"Output:    {OUTPUT_DIR}")
    print()

    class_names = load_names_from_yaml()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    row_csv = OUTPUT_DIR / "label_rows_analysis.csv"
    image_csv = OUTPUT_DIR / "image_summary.csv"

    image_rows = []
    class_counts = Counter()
    split_class_counts = defaultdict(Counter)
    crop_counts = defaultdict(int)

    total_images = 0
    missing_labels = 0
    total_valid_rows = 0
    total_bad_rows = 0
    total_yolo_valid = 0
    total_norm8_valid = 0
    coords = []

    with row_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "split",
                "image",
                "label",
                "label_exists",
                "line_no",
                "raw",
                "ok",
                "error",
                "class_id",
                "class_name_norm8_yaml",
                "raw23_reference_name",
                "x",
                "y",
                "w",
                "h",
                "valid_bbox_yolo_norm",
                "valid_class_norm8",
                "valid_class_raw23",
                "n_cols",
            ],
        )
        writer.writeheader()

        for split in SPLITS:
            images = list_images(split)
            print(f"{split}: {len(images)} images")

            for idx, img_path in enumerate(images, start=1):
                total_images += 1

                label_path = LABELS_ROOT / split / f"{img_path.stem}.txt"

                if not label_path.exists():
                    missing_labels += 1

                rows = read_label_file(label_path)

                render_ok = render_image(
                    img_path,
                    rows,
                    class_names,
                    OUTPUT_DIR / "renders_yolo_norm8" / split / f"{img_path.stem}.jpg",
                )

                saved_crops = save_object_crops(img_path, rows, class_names, split, crop_counts)

                valid_this = 0
                bad_this = 0

                for r in rows:
                    if not r.get("ok"):
                        total_bad_rows += 1
                        bad_this += 1

                        writer.writerow({
                            "split": split,
                            "image": str(img_path),
                            "label": str(label_path),
                            "label_exists": int(label_path.exists()),
                            "line_no": r.get("line_no"),
                            "raw": r.get("raw"),
                            "ok": 0,
                            "error": r.get("error"),
                        })
                        continue

                    total_valid_rows += 1

                    cid = r["class_id"]
                    class_counts[cid] += 1
                    split_class_counts[split][cid] += 1

                    coords.extend([r["x"], r["y"], r["w"], r["h"]])

                    if r["valid_bbox_yolo_norm"]:
                        total_yolo_valid += 1
                        valid_this += 1

                    if r["valid_class_norm8"]:
                        total_norm8_valid += 1

                    writer.writerow({
                        "split": split,
                        "image": str(img_path),
                        "label": str(label_path),
                        "label_exists": int(label_path.exists()),
                        "line_no": r["line_no"],
                        "raw": r["raw"],
                        "ok": 1,
                        "error": "",
                        "class_id": cid,
                        "class_name_norm8_yaml": class_names.get(cid, f"class_{cid}"),
                        "raw23_reference_name": RAW23_REFERENCE_NAMES.get(cid, ""),
                        "x": r["x"],
                        "y": r["y"],
                        "w": r["w"],
                        "h": r["h"],
                        "valid_bbox_yolo_norm": int(r["valid_bbox_yolo_norm"]),
                        "valid_class_norm8": int(r["valid_class_norm8"]),
                        "valid_class_raw23": int(r["valid_class_raw23"]),
                        "n_cols": r["n_cols"],
                    })

                image_rows.append({
                    "split": split,
                    "image": str(img_path),
                    "label": str(label_path),
                    "label_exists": int(label_path.exists()),
                    "label_rows": len(rows),
                    "valid_yolo_rows": valid_this,
                    "bad_rows": bad_this,
                    "rendered": int(render_ok),
                    "saved_crops": saved_crops,
                })

            print(f"Finished {split}")

    with image_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image",
                "label",
                "label_exists",
                "label_rows",
                "valid_yolo_rows",
                "bad_rows",
                "rendered",
                "saved_crops",
            ],
        )
        writer.writeheader()
        writer.writerows(image_rows)

    coords_np = np.array(coords, dtype=float) if coords else np.array([], dtype=float)

    classes = sorted(class_counts.keys())

    if classes and all(0 <= c <= 7 for c in classes):
        schema_guess = "NORM8 class schema"
    elif classes and all(0 <= c <= 22 for c in classes):
        schema_guess = "RAW 23-class schema or mixed"
    else:
        schema_guess = "UNKNOWN"

    yolo_ratio = total_yolo_valid / total_valid_rows if total_valid_rows else 0.0
    norm8_ratio = total_norm8_valid / total_valid_rows if total_valid_rows else 0.0

    if yolo_ratio > 0.98:
        geometry_guess = "YOLO_NORMALIZED_CENTER_XYWH"
    else:
        geometry_guess = "NOT_CLEAN_YOLO_OR_MIXED"

    report_path = OUTPUT_DIR / "annotation_format_report_Jan26_norm8.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("JAN-26 YOLO_norm8 annotation format diagnosis\n")
        f.write("=" * 80 + "\n")
        f.write(f"YOLO root:    {YOLO_ROOT}\n")
        f.write(f"Images root:  {IMAGES_ROOT}\n")
        f.write(f"Labels root:  {LABELS_ROOT}\n")
        f.write(f"YAML:         {DATASET_YAML}\n")
        f.write(f"Output dir:   {OUTPUT_DIR}\n")
        f.write("\n")
        f.write(f"Total images scanned:                 {total_images}\n")
        f.write(f"Missing label files:                  {missing_labels}\n")
        f.write(f"Valid parsed label rows:              {total_valid_rows}\n")
        f.write(f"Bad label rows:                       {total_bad_rows}\n")
        f.write(f"Rows valid YOLO normalized xywh:      {total_yolo_valid}/{total_valid_rows} ({yolo_ratio*100:.2f}%)\n")
        f.write(f"Rows with class ID in 0..7:           {total_norm8_valid}/{total_valid_rows} ({norm8_ratio*100:.2f}%)\n")
        f.write("\n")
        f.write(f"Class schema guess: {schema_guess}\n")
        f.write(f"Geometry guess:     {geometry_guess}\n")
        f.write("\n")
        f.write("Class names from dataset_norm8.yaml:\n")
        for cid in sorted(class_names):
            f.write(f"  {cid:2d}: {class_names[cid]}\n")
        f.write("\n")
        f.write("Class counts total:\n")
        for cid, count in sorted(class_counts.items()):
            cname = class_names.get(cid, f"class_{cid}")
            raw_ref = RAW23_REFERENCE_NAMES.get(cid, "")
            f.write(f"  {cid:2d}: {cname:20s} count={count:6d} raw23_reference_if_raw23={raw_ref}\n")
        f.write("\n")
        f.write("Class counts by split:\n")
        for split in SPLITS:
            f.write(f"  [{split}]\n")
            for cid in sorted(class_names):
                f.write(f"    {cid:2d}: {class_names[cid]:20s} {split_class_counts[split][cid]}\n")
        if coords_np.size:
            f.write("\nCoordinate value statistics over x,y,w,h:\n")
            f.write(f"  min:    {coords_np.min():.6f}\n")
            f.write(f"  max:    {coords_np.max():.6f}\n")
            f.write(f"  mean:   {coords_np.mean():.6f}\n")
            f.write(f"  median: {np.median(coords_np):.6f}\n")
        f.write("\nImportant comparison:\n")
        f.write("  Jan-26 YOLO_norm8 should use:\n")
        for cid in range(8):
            f.write(f"    {cid}: {EXPECTED_NORM8_NAMES[cid]}\n")
        f.write("\n")
        f.write("  Jun-26 raw labels should use:\n")
        f.write("    1=Container, 4=Papier, 6=Plastiksack, 9=BAW-Sack, 22=Wagon\n")
        f.write("\n")
        f.write("Conclusion rule:\n")
        f.write("  Use class 0 as BAW-Sack only in Jan-26 YOLO_norm8 or in already-created binary D2 outputs.\n")
        f.write("  Use class 9 as BAW-Sack when reading raw Jun-26 labels.\n")
        f.write("\nOutputs:\n")
        f.write(f"  Rendered labels: {OUTPUT_DIR / 'renders_yolo_norm8'}\n")
        f.write(f"  Crops by class:  {OUTPUT_DIR / 'crops_by_class'}\n")
        f.write(f"  Row CSV:         {row_csv}\n")
        f.write(f"  Image CSV:       {image_csv}\n")

    print()
    print("=" * 100)
    print("JAN-26 YOLO_norm8 DIAGNOSIS DONE")
    print("=" * 100)
    print(f"Report:")
    print(f"  {report_path}")
    print(f"Rendered labels:")
    print(f"  {OUTPUT_DIR / 'renders_yolo_norm8'}")
    print(f"Crops by class:")
    print(f"  {OUTPUT_DIR / 'crops_by_class'}")
    print(f"Row CSV:")
    print(f"  {row_csv}")
    print(f"Image CSV:")
    print(f"  {image_csv}")
    print("=" * 100)
    print(f"Class schema guess: {schema_guess}")
    print(f"Geometry guess:     {geometry_guess}")
    print("=" * 100)


if __name__ == "__main__":
    main()
