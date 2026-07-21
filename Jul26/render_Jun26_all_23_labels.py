
from pathlib import Path
import csv
import math
import random

# ============================================================
# RENDER JUN-26 RAW DATASET — ALL IMAGES, ALL 23 LABELS
# ============================================================
#
# Purpose:
#   Diagnose whether YOLO labels are read/interpreted correctly.
#
# What this script does:
#   1. Reads every image in:
#        C:\Hadi\Foreign_Material\Data\Ours\Jun-26\images
#   2. Reads matching YOLO labels from:
#        C:\Hadi\Foreign_Material\Data\Ours\Jun-26\labels
#   3. Renders all label boxes, not only BAW-Sack.
#   4. Creates:
#        rendered_cv2_no_exif\
#        rendered_pil_exif_transposed\
#        crops_by_class\
#        render_summary.csv
#
# Why two rendering modes?
#   If original images contain EXIF orientation metadata, PIL and OpenCV may
#   interpret image orientation differently. YOLO/OpenCV usually ignores EXIF
#   orientation, while some annotation tools/viewers may apply it.
#
#   If rendered_cv2_no_exif is aligned, training reads labels correctly.
#   If rendered_pil_exif_transposed is aligned but cv2 is wrong, there is an
#   EXIF/orientation problem.
#
# ============================================================

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


SOURCE_DATASET = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26")

IMAGES_DIR = SOURCE_DATASET / "images"
LABELS_DIR = SOURCE_DATASET / "labels"
DATA_YAML = SOURCE_DATASET / "data.yaml"

OUTPUT_DIR = SOURCE_DATASET / "_rendered_all_23_labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Set True if you want individual object crops too.
SAVE_OBJECT_CROPS = True

# Limit crops per class to avoid too many files. Set None for all.
MAX_CROPS_PER_CLASS = 300

# Label font size
FONT_SIZE = 18

# Fallback class names for original 23-class dataset.
# If data.yaml has names, those names will be used instead.
FALLBACK_CLASS_NAMES = {
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


def load_class_names():
    """
    Minimal data.yaml parser for names.
    Supports:
      names:
        0: name
        1: name
    and
      names: [a, b, c]
    """
    if not DATA_YAML.exists():
        return dict(FALLBACK_CLASS_NAMES)

    txt = DATA_YAML.read_text(encoding="utf-8", errors="ignore").splitlines()

    names = {}
    in_names = False

    for line in txt:
        raw = line.rstrip()
        stripped = raw.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("names:"):
            after = stripped[len("names:"):].strip()
            if after.startswith("[") and after.endswith("]"):
                items = after.strip("[]").split(",")
                for i, item in enumerate(items):
                    names[i] = item.strip().strip("'\"")
                return names if names else dict(FALLBACK_CLASS_NAMES)
            in_names = True
            continue

        if in_names:
            if not raw.startswith(" ") and not raw.startswith("\t"):
                break
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                try:
                    cid = int(k.strip())
                    cname = v.strip().strip("'\"")
                    names[cid] = cname
                except ValueError:
                    pass

    if names:
        # Ensure all 0..22 exist.
        for cid in range(23):
            names.setdefault(cid, FALLBACK_CLASS_NAMES.get(cid, f"class_{cid}"))
        return names

    return dict(FALLBACK_CLASS_NAMES)


CLASS_NAMES = load_class_names()


def color_for_class(cid):
    # deterministic bright-ish colors
    rng = random.Random(cid * 1009 + 17)
    return (
        rng.randint(40, 255),
        rng.randint(40, 255),
        rng.randint(40, 255),
    )


def find_images():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    return sorted([
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def read_label_file(label_path):
    rows = []

    if not label_path.exists():
        return rows

    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                rows.append({
                    "valid": False,
                    "error": "malformed",
                    "line_number": line_number,
                    "raw": line,
                })
                continue

            try:
                cid = int(float(parts[0]))
                x, y, w, h = [float(v) for v in parts[1:5]]
            except ValueError:
                rows.append({
                    "valid": False,
                    "error": "numeric_error",
                    "line_number": line_number,
                    "raw": line,
                })
                continue

            valid_bbox = 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
            valid_class = 0 <= cid <= 22

            rows.append({
                "valid": valid_bbox and valid_class,
                "valid_bbox": valid_bbox,
                "valid_class": valid_class,
                "class_id": cid,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "line_number": line_number,
                "raw": line,
            })

    return rows


def yolo_to_xyxy_px(row, img_w, img_h):
    x = row["x"]
    y = row["y"]
    w = row["w"]
    h = row["h"]

    x1 = int(round((x - w / 2) * img_w))
    y1 = int(round((y - h / 2) * img_h))
    x2 = int(round((x + w / 2) * img_w))
    y2 = int(round((y + h / 2) * img_h))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(1, min(img_w, x2))
    y2 = max(1, min(img_h, y2))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    return x1, y1, x2, y2


def get_font():
    try:
        return ImageFont.truetype("arial.ttf", FONT_SIZE)
    except Exception:
        return ImageFont.load_default()


def draw_labels_pil(img, rows, mode_name):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = get_font()
    img_w, img_h = img.size

    for row in rows:
        if not row.get("valid", False):
            continue

        cid = row["class_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        color = color_for_class(cid)

        x1, y1, x2, y2 = yolo_to_xyxy_px(row, img_w, img_h)

        width = max(2, int(round(min(img_w, img_h) / 250)))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        label = f"{cid}:{cname}"
        text_x = x1
        text_y = max(0, y1 - FONT_SIZE - 4)

        # text background
        try:
            bbox = draw.textbbox((text_x, text_y), label, font=font)
            draw.rectangle(bbox, fill=color)
        except Exception:
            draw.rectangle([text_x, text_y, text_x + len(label) * 8, text_y + FONT_SIZE + 4], fill=color)

        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    # Add mode tag
    tag = mode_name
    draw.rectangle([0, 0, min(img_w, 520), 28], fill=(255, 255, 255))
    draw.text((4, 4), tag, fill=(0, 0, 0), font=font)

    return img


def render_cv2_no_exif(image_path, rows, out_path):
    if not HAS_CV2:
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    img_h, img_w = img.shape[:2]

    for row in rows:
        if not row.get("valid", False):
            continue

        cid = row["class_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}")
        color_rgb = color_for_class(cid)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        x1, y1, x2, y2 = yolo_to_xyxy_px(row, img_w, img_h)

        thickness = max(2, int(round(min(img_w, img_h) / 250)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, thickness)

        label = f"{cid}:{cname}"
        ty = max(15, y1 - 5)
        cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

    cv2.putText(img, "cv2_no_exif_orientation", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, "cv2_no_exif_orientation", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def save_object_crops(img_cv2_like, image_path, rows, crop_counts):
    """
    Save crops based on cv2/no-EXIF dimensions, because that is closest to YOLO training.
    """
    if not SAVE_OBJECT_CROPS:
        return []

    saved = []

    img = img_cv2_like.convert("RGB")
    img_w, img_h = img.size

    for row in rows:
        if not row.get("valid", False):
            continue

        cid = row["class_id"]
        cname = CLASS_NAMES.get(cid, f"class_{cid}").replace("/", "_").replace("\\", "_").replace(" ", "_")

        if MAX_CROPS_PER_CLASS is not None and crop_counts[cid] >= MAX_CROPS_PER_CLASS:
            continue

        x1, y1, x2, y2 = yolo_to_xyxy_px(row, img_w, img_h)

        crop = img.crop((x1, y1, x2, y2))
        class_dir = OUTPUT_DIR / "crops_by_class" / f"{cid:02d}_{cname}"
        class_dir.mkdir(parents=True, exist_ok=True)

        out_crop = class_dir / f"{image_path.stem}_line{row['line_number']:03d}.jpg"
        crop.save(out_crop, quality=95)

        crop_counts[cid] += 1
        saved.append(out_crop)

    return saved


def main():
    print("=" * 100)
    print("Rendering all Jun-26 images with all 23 labels")
    print("=" * 100)
    print(f"Source dataset: {SOURCE_DATASET}")
    print(f"Images:         {IMAGES_DIR}")
    print(f"Labels:         {LABELS_DIR}")
    print(f"Output:         {OUTPUT_DIR}")
    print()

    if OUTPUT_DIR.exists():
        # Do not delete automatically; safer. We overwrite same filenames.
        pass

    (OUTPUT_DIR / "rendered_cv2_no_exif").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "rendered_pil_exif_transposed").mkdir(parents=True, exist_ok=True)

    images = find_images()
    crop_counts = {cid: 0 for cid in range(23)}

    summary_rows = []

    for idx, image_path in enumerate(images, start=1):
        label_path = LABELS_DIR / f"{image_path.stem}.txt"
        rows = read_label_file(label_path)

        valid_rows = [r for r in rows if r.get("valid", False)]
        invalid_rows = [r for r in rows if not r.get("valid", False)]

        class_counts = {}
        for r in valid_rows:
            cid = r["class_id"]
            class_counts[cid] = class_counts.get(cid, 0) + 1

        # 1) cv2/no-EXIF rendering: closest to YOLO/OpenCV training
        out_cv2 = OUTPUT_DIR / "rendered_cv2_no_exif" / image_path.name
        cv2_ok = render_cv2_no_exif(image_path, rows, out_cv2)

        # Also create PIL no-EXIF image for crops.
        pil_no_exif = Image.open(image_path).convert("RGB")

        # 2) PIL EXIF-transposed rendering: detects orientation mismatch
        pil_exif = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        rendered_pil = draw_labels_pil(pil_exif, rows, "pil_exif_transposed")
        out_pil = OUTPUT_DIR / "rendered_pil_exif_transposed" / image_path.name
        out_pil.parent.mkdir(parents=True, exist_ok=True)
        rendered_pil.save(out_pil, quality=95)

        # Crops based on no-EXIF dimensions
        saved_crops = save_object_crops(pil_no_exif, image_path, rows, crop_counts)

        summary_rows.append({
            "image": str(image_path),
            "label": str(label_path),
            "label_exists": int(label_path.exists()),
            "total_lines": len(rows),
            "valid_objects": len(valid_rows),
            "invalid_lines": len(invalid_rows),
            "cv2_rendered": int(cv2_ok),
            "cv2_render_path": str(out_cv2),
            "pil_exif_render_path": str(out_pil),
            "saved_crops": len(saved_crops),
            **{f"class_{cid}": class_counts.get(cid, 0) for cid in range(23)}
        })

        if idx % 100 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)} images...")

    summary_csv = OUTPUT_DIR / "render_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image", "label", "label_exists",
            "total_lines", "valid_objects", "invalid_lines",
            "cv2_rendered", "cv2_render_path", "pil_exif_render_path",
            "saved_crops",
        ] + [f"class_{cid}" for cid in range(23)]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Rendered cv2/no-EXIF images:")
    print(f"  {OUTPUT_DIR / 'rendered_cv2_no_exif'}")
    print(f"Rendered PIL/EXIF-transposed images:")
    print(f"  {OUTPUT_DIR / 'rendered_pil_exif_transposed'}")
    print(f"Object crops by class:")
    print(f"  {OUTPUT_DIR / 'crops_by_class'}")
    print(f"Summary CSV:")
    print(f"  {summary_csv}")
    print()
    print("How to diagnose:")
    print("  1. First inspect rendered_cv2_no_exif. This is closest to YOLO training.")
    print("  2. If boxes are wrong there but correct in rendered_pil_exif_transposed,")
    print("     then the issue is EXIF/orientation handling.")
    print("  3. Inspect crops_by_class/09_BAW-Sack or similar class folder.")
    print("     If crops are wrong, D2 crop-verifier dataset generation must be fixed.")
    print("=" * 100)


if __name__ == "__main__":
    main()
