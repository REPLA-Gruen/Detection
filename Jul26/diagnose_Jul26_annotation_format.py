
from pathlib import Path
import csv
import json
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

# ============================================================
# DIAGNOSE JUL-26 ANNOTATION FORMAT AND CLASS SCHEMA
# ============================================================
#
# Purpose:
#   Check whether C:\Hadi\Foreign_Material\Data\Ours\Jul-26\labels
#   is interpreted correctly.
#
# It tests and renders four possible interpretations of each label row:
#
#   A) YOLO normalized center xywh:
#      class x_center y_center width height, all coordinates 0..1
#
#   B) Label-Studio-style percent top-left xywh:
#      class x_top_left y_top_left width height, coordinates 0..100
#
#   C) Absolute-pixel top-left xywh:
#      class x_top_left y_top_left width height, coordinates in pixels
#
#   D) Absolute-pixel center xywh:
#      class x_center y_center width height, coordinates in pixels
#
# Outputs:
#   C:\Hadi\Foreign_Material\Data\Ours\Jul-26\_annotation_format_diagnosis
#
# Inspect:
#   renders_compare
#   crops_yolo_norm_center_by_class
#   annotation_format_report.txt
#   label_rows_analysis.csv
#
# ============================================================

SOURCE_DATASET = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jul-26")
IMAGES_DIR = SOURCE_DATASET / "images"
LABELS_DIR = SOURCE_DATASET / "labels"
DATA_YAML = SOURCE_DATASET / "data.yaml"
if not DATA_YAML.exists():
    DATA_YAML = SOURCE_DATASET / "data.yml"

OUT_DIR = SOURCE_DATASET / "_annotation_format_diagnosis"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Render all images. Set a number like 200 if you want a faster first run.
MAX_IMAGES_TO_RENDER = None

# Save crops for YOLO normalized interpretation, because that is what YOLO training expects.
SAVE_CROPS_YOLO_NORM = True
MAX_CROPS_PER_CLASS = 250

RANDOM_SEED = 42


FALLBACK_NAMES_23 = {
    0: "class_0",
    1: "Container",
    2: "class_2",
    3: "class_3",
    4: "Papier",
    5: "class_5",
    6: "Plastiksack",
    7: "class_7",
    8: "class_8",
    9: "BAW-Sack",
    10: "class_10",
    11: "class_11",
    12: "class_12",
    13: "class_13",
    14: "class_14",
    15: "class_15",
    16: "class_16",
    17: "class_17",
    18: "class_18",
    19: "class_19",
    20: "class_20",
    21: "class_21",
    22: "Wagon",
}

NORM8_NAMES = {
    0: "BAW-Sack",
    1: "Plastic",
    2: "Metal",
    3: "Glass",
    4: "Paper_Cardboard",
    5: "Other",
    6: "Wagon",
    7: "Container",
}


def load_yaml_names():
    if not DATA_YAML.exists():
        return {}

    lines = DATA_YAML.read_text(encoding="utf-8", errors="ignore").splitlines()
    names = {}
    in_names = False

    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue

        if s.startswith("names:"):
            after = s[len("names:"):].strip()
            if after.startswith("[") and after.endswith("]"):
                items = [x.strip().strip("'\"") for x in after.strip("[]").split(",")]
                return {i: v for i, v in enumerate(items)}
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

    return names


def class_name(cid, yaml_names):
    if cid in yaml_names:
        return yaml_names[cid]
    if cid in FALLBACK_NAMES_23:
        return FALLBACK_NAMES_23[cid]
    return f"class_{cid}"


def color(cid):
    rng = random.Random(cid * 99991 + 123)
    return (rng.randint(40, 255), rng.randint(40, 255), rng.randint(40, 255))


def list_images():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    imgs = sorted(p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if MAX_IMAGES_TO_RENDER is not None:
        imgs = imgs[:MAX_IMAGES_TO_RENDER]
    return imgs


def read_rows(label_path):
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
                rows.append({"ok": False, "error": "less_than_5_columns", "line_no": line_no, "raw": raw})
                continue
            try:
                cid = int(float(parts[0]))
                vals = [float(x) for x in parts[1:5]]
            except ValueError:
                rows.append({"ok": False, "error": "numeric_error", "line_no": line_no, "raw": raw})
                continue
            rows.append({
                "ok": True,
                "line_no": line_no,
                "raw": raw,
                "class_id": cid,
                "a": vals[0],
                "b": vals[1],
                "c": vals[2],
                "d": vals[3],
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


def box_yolo_norm_center(row, W, H):
    x, y, w, h = row["a"], row["b"], row["c"], row["d"]
    return clip_xyxy(((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H), W, H)


def box_ls_percent_topleft(row, W, H):
    # class x y width height where x/y/width/height are percent 0..100 of current image.
    x, y, w, h = row["a"], row["b"], row["c"], row["d"]
    return clip_xyxy((x / 100 * W, y / 100 * H, (x + w) / 100 * W, (y + h) / 100 * H), W, H)


def box_abs_topleft(row, W, H):
    x, y, w, h = row["a"], row["b"], row["c"], row["d"]
    return clip_xyxy((x, y, x + w, y + h), W, H)


def box_abs_center(row, W, H):
    x, y, w, h = row["a"], row["b"], row["c"], row["d"]
    return clip_xyxy((x - w / 2, y - h / 2, x + w / 2, y + h / 2), W, H)


INTERPRETATIONS = [
    ("A_yolo_norm_center", box_yolo_norm_center),
    ("B_ls_percent_topleft", box_ls_percent_topleft),
    ("C_abs_topleft", box_abs_topleft),
    ("D_abs_center", box_abs_center),
]


def draw_interpretation(img, rows, title, box_fn, yaml_names):
    vis = img.copy()
    H, W = vis.shape[:2]

    cv2.rectangle(vis, (0, 0), (min(W, 900), 34), (255, 255, 255), -1)
    cv2.putText(vis, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    for r in rows:
        if not r.get("ok"):
            continue
        cid = r["class_id"]
        x1, y1, x2, y2 = box_fn(r, W, H)
        col = color(cid)
        col_bgr = (col[2], col[1], col[0])
        thick = max(2, int(round(min(W, H) / 250)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col_bgr, thick)
        label = f"{cid}:{class_name(cid, yaml_names)}"
        cv2.putText(vis, label, (x1 + 3, max(18, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_bgr, 2, cv2.LINE_AA)

    return vis


def make_grid(images):
    # 2x2 grid, resize panels to same size
    h0, w0 = images[0].shape[:2]
    panel_w = 640
    panel_h = int(round(panel_w * h0 / w0))
    resized = [cv2.resize(im, (panel_w, panel_h)) for im in images]
    top = np.hstack([resized[0], resized[1]])
    bottom = np.hstack([resized[2], resized[3]])
    return np.vstack([top, bottom])


def valid_yolo_norm(row):
    if not row.get("ok"):
        return False
    x, y, w, h = row["a"], row["b"], row["c"], row["d"]
    return 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1


def valid_ls_percent(row):
    if not row.get("ok"):
        return False
    x, y, w, h = row["a"], row["b"], row["c"], row["d"]
    return 0 <= x <= 100 and 0 <= y <= 100 and 0 < w <= 100 and 0 < h <= 100 and x + w <= 110 and y + h <= 110


def save_yolo_crops(img, img_path, rows, yaml_names, crop_counts):
    if not SAVE_CROPS_YOLO_NORM:
        return 0

    H, W = img.shape[:2]
    saved = 0

    for r in rows:
        if not r.get("ok") or not valid_yolo_norm(r):
            continue

        cid = r["class_id"]
        if MAX_CROPS_PER_CLASS is not None and crop_counts[cid] >= MAX_CROPS_PER_CLASS:
            continue

        x1, y1, x2, y2 = box_yolo_norm_center(r, W, H)

        cname = class_name(cid, yaml_names).replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_dir = OUT_DIR / "crops_yolo_norm_center_by_class" / f"{cid:02d}_{cname}"
        out_dir.mkdir(parents=True, exist_ok=True)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        out_path = out_dir / f"{img_path.stem}_line{r['line_no']:03d}.jpg"
        cv2.imwrite(str(out_path), crop)

        crop_counts[cid] += 1
        saved += 1

    return saved


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "renders_compare").mkdir(parents=True, exist_ok=True)

    yaml_names = load_yaml_names()

    images = list_images()

    all_values = []
    class_counts = Counter()
    n_lines = 0
    n_bad = 0
    n_yolo_valid = 0
    n_ls_valid = 0
    n_coord_gt1 = 0
    n_coord_gt100 = 0
    n_labels_missing = 0
    n_images_with_labels = 0

    row_csv = OUT_DIR / "label_rows_analysis.csv"
    crop_counts = defaultdict(int)

    with row_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "image", "label", "line_no", "raw", "ok", "error",
                "class_id", "class_name", "a", "b", "c", "d",
                "valid_yolo_norm_center", "valid_ls_percent_topleft",
                "coords_max", "n_cols",
            ],
        )
        writer.writeheader()

        for idx, img_path in enumerate(images, start=1):
            label_path = LABELS_DIR / f"{img_path.stem}.txt"
            rows = read_rows(label_path)

            if not label_path.exists():
                n_labels_missing += 1
            if rows:
                n_images_with_labels += 1

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            panels = []
            for title, fn in INTERPRETATIONS:
                panels.append(draw_interpretation(img, rows, title, fn, yaml_names))

            grid = make_grid(panels)
            cv2.imwrite(str(OUT_DIR / "renders_compare" / f"{img_path.stem}.jpg"), grid)

            saved_crops = save_yolo_crops(img, img_path, rows, yaml_names, crop_counts)

            for r in rows:
                if not r.get("ok"):
                    n_bad += 1
                    writer.writerow({
                        "image": str(img_path),
                        "label": str(label_path),
                        "line_no": r.get("line_no"),
                        "raw": r.get("raw"),
                        "ok": 0,
                        "error": r.get("error"),
                    })
                    continue

                n_lines += 1
                cid = r["class_id"]
                class_counts[cid] += 1
                vals = [r["a"], r["b"], r["c"], r["d"]]
                all_values.extend(vals)

                if max(vals) > 1:
                    n_coord_gt1 += 1
                if max(vals) > 100:
                    n_coord_gt100 += 1

                vy = valid_yolo_norm(r)
                vl = valid_ls_percent(r)

                if vy:
                    n_yolo_valid += 1
                if vl:
                    n_ls_valid += 1

                writer.writerow({
                    "image": str(img_path),
                    "label": str(label_path),
                    "line_no": r["line_no"],
                    "raw": r["raw"],
                    "ok": 1,
                    "error": "",
                    "class_id": cid,
                    "class_name": class_name(cid, yaml_names),
                    "a": r["a"],
                    "b": r["b"],
                    "c": r["c"],
                    "d": r["d"],
                    "valid_yolo_norm_center": int(vy),
                    "valid_ls_percent_topleft": int(vl),
                    "coords_max": max(vals),
                    "n_cols": r["n_cols"],
                })

            if idx % 100 == 0 or idx == len(images):
                print(f"Processed {idx}/{len(images)} images")

    values_np = np.array(all_values, dtype=float) if all_values else np.array([], dtype=float)

    if len(class_counts) == 0:
        schema_guess = "NO_LABEL_ROWS_FOUND"
    else:
        classes = sorted(class_counts)
        if all(0 <= c <= 7 for c in classes):
            schema_guess = "Likely NORM8 class schema: 0=BAW-Sack, 1=Plastic, ..., 7=Container"
        elif all(0 <= c <= 22 for c in classes):
            schema_guess = "Likely original/raw 23-class schema, but class names must come from data.yaml or mapping"
        else:
            schema_guess = "Unknown class schema; class IDs outside 0..22 found"

    if n_lines > 0:
        yolo_ratio = n_yolo_valid / n_lines
        ls_ratio = n_ls_valid / n_lines
    else:
        yolo_ratio = 0
        ls_ratio = 0

    if yolo_ratio > 0.98 and n_coord_gt1 == 0:
        geom_guess = "YOLO_NORMALIZED_CENTER_XYWH"
    elif ls_ratio > 0.98 and n_coord_gt1 > 0 and n_coord_gt100 == 0:
        geom_guess = "LIKELY_LABEL_STUDIO_PERCENT_TOPLEFT_XYWH_OR_PERCENT_FORMAT"
    elif n_coord_gt100 > 0:
        geom_guess = "LIKELY_ABSOLUTE_PIXEL_COORDINATES_OR_MIXED_FORMAT"
    else:
        geom_guess = "MIXED_OR_UNKNOWN"

    report = OUT_DIR / "annotation_format_report.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("JUL-26 annotation format diagnosis\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {SOURCE_DATASET}\n")
        f.write(f"Images dir: {IMAGES_DIR}\n")
        f.write(f"Labels dir: {LABELS_DIR}\n")
        f.write(f"data.yaml: {DATA_YAML if DATA_YAML.exists() else '(not found)'}\n")
        f.write("\n")
        f.write(f"Images scanned: {len(images)}\n")
        f.write(f"Images with label rows: {n_images_with_labels}\n")
        f.write(f"Images missing label file: {n_labels_missing}\n")
        f.write(f"Valid label rows: {n_lines}\n")
        f.write(f"Bad label rows: {n_bad}\n")
        f.write("\n")
        f.write("Class counts:\n")
        for cid, count in sorted(class_counts.items()):
            f.write(f"  {cid:3d}: {class_name(cid, yaml_names):25s} {count}\n")
        f.write("\n")
        f.write(f"Class schema guess: {schema_guess}\n")
        f.write(f"Geometry guess:     {geom_guess}\n")
        f.write("\n")
        f.write(f"Rows valid as YOLO normalized center xywh:      {n_yolo_valid}/{n_lines} ({yolo_ratio*100:.2f}%)\n")
        f.write(f"Rows valid as LS percent top-left xywh:         {n_ls_valid}/{n_lines} ({ls_ratio*100:.2f}%)\n")
        f.write(f"Rows with any coordinate > 1:                   {n_coord_gt1}/{n_lines}\n")
        f.write(f"Rows with any coordinate > 100:                 {n_coord_gt100}/{n_lines}\n")
        if values_np.size:
            f.write("\nCoordinate value statistics over columns 2-5:\n")
            f.write(f"  min:    {float(values_np.min()):.6f}\n")
            f.write(f"  max:    {float(values_np.max()):.6f}\n")
            f.write(f"  mean:   {float(values_np.mean()):.6f}\n")
            f.write(f"  median: {float(np.median(values_np)):.6f}\n")
        f.write("\nOutputs:\n")
        f.write(f"  Compare renders: {OUT_DIR / 'renders_compare'}\n")
        f.write(f"  YOLO-normalized crops by class: {OUT_DIR / 'crops_yolo_norm_center_by_class'}\n")
        f.write(f"  Row CSV: {row_csv}\n")
        f.write("\nHow to decide:\n")
        f.write("  Open renders_compare. Each image has 4 panels:\n")
        f.write("    A = YOLO normalized center xywh\n")
        f.write("    B = Label Studio percent top-left xywh\n")
        f.write("    C = absolute pixel top-left xywh\n")
        f.write("    D = absolute pixel center xywh\n")
        f.write("  The panel where boxes match the objects shows the real format.\n")
        f.write("  If A is correct, YOLO training reads labels correctly.\n")
        f.write("  If B is correct, the labels are not YOLO labels and must be converted first.\n")
        f.write("  If class schema guess is NORM8 but D2 scripts expect raw IDs 9/6/4/1/22, class mapping is wrong.\n")

    print("=" * 100)
    print("ANNOTATION FORMAT DIAGNOSIS DONE")
    print("=" * 100)
    print(f"Report:")
    print(f"  {report}")
    print(f"Compare renders:")
    print(f"  {OUT_DIR / 'renders_compare'}")
    print(f"YOLO-normalized crops by class:")
    print(f"  {OUT_DIR / 'crops_yolo_norm_center_by_class'}")
    print(f"CSV:")
    print(f"  {row_csv}")
    print("=" * 100)
    print(f"Geometry guess: {geom_guess}")
    print(f"Class schema guess: {schema_guess}")
    print("=" * 100)


if __name__ == "__main__":
    main()
