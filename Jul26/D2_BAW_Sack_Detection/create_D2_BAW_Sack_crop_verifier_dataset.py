
from pathlib import Path
import random, csv, shutil
from collections import Counter
from PIL import Image

# ============================================================
# CREATE D2 BAW-SACK CROP VERIFIER DATASET
# ============================================================
# Creates a classification dataset for the second AI stage:
# BAW_Sack vs Not_BAW_Sack.
#
# Positive crops come from GT BAW-Sack boxes.
# Negative crops come from background/non-overlapping regions.
# Optional hard negatives can be added later from weak detector false positives.
# ============================================================

SOURCE_D2_DATASET = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Focused_Datasets_With_Limited_Negatives_Optimized_80_10_10\D2_BAW_Sack_binary_neg1to1")
OUTPUT_DATASET = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26_D2_BAW_Sack_AI_Module\D2_BAW_Sack_crop_verifier")

SPLITS = ["train", "val", "test"]
CLASS_ID = 0
POS_CLASS = "BAW_Sack"
NEG_CLASS = "Not_BAW_Sack"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

RANDOM_SEED = 42
SAVE_CROP_SIZE = 384
POSITIVE_PADDING_FACTORS = [0.00, 0.15, 0.30]
NEGATIVE_RANDOM_CROPS_PER_IMAGE = 3
RANDOM_NEG_MAX_IOU_WITH_GT = 0.03
RANDOM_NEG_TRIALS_PER_CROP = 60
MIN_CROP_FRAC = 0.06
MAX_CROP_FRAC = 0.45


def reset_output():
    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)
    for split in SPLITS:
        (OUTPUT_DATASET / split / POS_CLASS).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DATASET / split / NEG_CLASS).mkdir(parents=True, exist_ok=True)


def images_for_split(split):
    d = SOURCE_D2_DATASET / "images" / split
    if not d.exists():
        raise FileNotFoundError(f"Missing image folder: {d}")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_yolo_boxes(label_path):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if cls == CLASS_ID:
            out.append((x, y, w, h))
    return out


def clip(box, W, H):
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


def yolo_to_px(b, W, H):
    x, y, w, h = b
    return clip(((x - w/2)*W, (y - h/2)*H, (x + w/2)*W, (y + h/2)*H), W, H)


def expand(box, W, H, pad):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return clip((x1 - bw*pad, y1 - bh*pad, x2 + bw*pad, y2 + bh*pad), W, H)


def area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def max_iou(box, gt):
    return max([iou(box, g) for g in gt], default=0.0)


def save_crop(img, box, path):
    crop = img.crop(box).resize((SAVE_CROP_SIZE, SAVE_CROP_SIZE))
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path, quality=95)


def random_negative_box(rng, W, H, gt_boxes):
    min_side = min(W, H)
    for _ in range(RANDOM_NEG_TRIALS_PER_CROP):
        cw = int(rng.uniform(MIN_CROP_FRAC, MAX_CROP_FRAC) * min_side)
        ch = int(cw * rng.uniform(0.6, 1.7))
        cw, ch = max(16, min(cw, W)), max(16, min(ch, H))
        x1 = rng.randint(0, max(0, W-cw))
        y1 = rng.randint(0, max(0, H-ch))
        box = (x1, y1, x1+cw, y1+ch)
        if max_iou(box, gt_boxes) <= RANDOM_NEG_MAX_IOU_WITH_GT:
            return box
    return None


def main():
    print("="*100)
    print("Creating D2 BAW-Sack crop verifier dataset")
    print("="*100)
    reset_output()
    rng = random.Random(RANDOM_SEED)
    rows, counts = [], Counter()

    for split in SPLITS:
        for img_path in images_for_split(split):
            label_path = SOURCE_D2_DATASET / "labels" / split / f"{img_path.stem}.txt"
            gt_yolo = read_yolo_boxes(label_path)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                W, H = img.size
                gt_px = [yolo_to_px(b, W, H) for b in gt_yolo]

                for gi, gt in enumerate(gt_px):
                    for pad in POSITIVE_PADDING_FACTORS:
                        box = expand(gt, W, H, pad)
                        outp = OUTPUT_DATASET / split / POS_CLASS / f"{img_path.stem}_gt{gi:03d}_pad{str(pad).replace('.','p')}.jpg"
                        save_crop(img, box, outp)
                        rows.append([split, POS_CLASS, str(img_path), str(outp), "positive_gt", *box])
                        counts[(split, POS_CLASS)] += 1

                for ni in range(NEGATIVE_RANDOM_CROPS_PER_IMAGE):
                    box = random_negative_box(rng, W, H, gt_px)
                    if box is None:
                        continue
                    outp = OUTPUT_DATASET / split / NEG_CLASS / f"{img_path.stem}_randneg{ni:03d}.jpg"
                    save_crop(img, box, outp)
                    rows.append([split, NEG_CLASS, str(img_path), str(outp), "random_negative", *box])
                    counts[(split, NEG_CLASS)] += 1

    manifest = OUTPUT_DATASET / "crop_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","class","source_image","crop_path","crop_type","x1","y1","x2","y2"])
        w.writerows(rows)

    print(f"Output:   {OUTPUT_DATASET}")
    print(f"Manifest: {manifest}")
    for split in SPLITS:
        pos = counts[(split, POS_CLASS)]
        neg = counts[(split, NEG_CLASS)]
        print(f"{split:5s}: {POS_CLASS}={pos:5d}, {NEG_CLASS}={neg:5d}, total={pos+neg:5d}")
    print("DONE")


if __name__ == "__main__":
    main()
