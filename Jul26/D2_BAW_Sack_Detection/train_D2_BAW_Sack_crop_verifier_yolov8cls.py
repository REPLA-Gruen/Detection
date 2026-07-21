
from pathlib import Path
from ultralytics import YOLO
import time, csv

# ============================================================
# TRAIN D2 BAW-SACK CROP VERIFIER CLASSIFIER
# ============================================================

DATASET_DIR = Path(r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26_D2_BAW_Sack_AI_Module\D2_BAW_Sack_crop_verifier")
RUNS_ROOT = Path(r"C:\Hadi\Foreign_Material\Runs\D2_BAW_Sack_AI_Module")
DATASET_RUN_NAME = "D2_BAW_Sack_crop_verifier_cls"

BASE_MODEL = "yolov8m-cls.pt"
IMG_SIZE = 384
EPOCHS = 100
PATIENCE = 20
BATCH = 16
WORKERS = 4
DEVICE = 0
SEED = 42
USE_PREVIOUS_BEST_FROM_THIS_DATASET = False


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def dataset_runs_dir():
    return RUNS_ROOT / DATASET_RUN_NAME


def check_dataset():
    for split in ["train", "val", "test"]:
        for cls in ["BAW_Sack", "Not_BAW_Sack"]:
            p = DATASET_DIR / split / cls
            if not p.exists():
                raise FileNotFoundError(f"Missing folder: {p}")


def find_previous_best():
    root = dataset_runs_dir()
    if not root.exists():
        return None
    c = []
    for p in root.rglob("weights/best.pt"):
        try:
            c.append((p.stat().st_mtime, p))
        except OSError:
            pass
    if not c:
        return None
    c.sort(reverse=True)
    return c[0][1]


def choose_initial_weight():
    if USE_PREVIOUS_BEST_FROM_THIS_DATASET:
        p = find_previous_best()
        if p is not None:
            return p, "previous crop verifier best.pt"
    return BASE_MODEL, "base YOLOv8 classification pretrained model"


def actual_save_dir(results, model, fallback):
    sd = getattr(results, "save_dir", None)
    if sd is None and hasattr(model, "trainer"):
        sd = getattr(model.trainer, "save_dir", None)
    return Path(sd) if sd is not None else fallback


def write_info(run_dir, init_weight, init_reason):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "training_info.txt").open("w", encoding="utf-8") as f:
        f.write("Module: D2 BAW-Sack crop verifier classifier\n")
        f.write(f"Dataset dir: {DATASET_DIR}\n")
        f.write(f"Initial weight: {init_weight}\n")
        f.write(f"Initial reason: {init_reason}\n")
        f.write(f"Base model: {BASE_MODEL}\n")
        f.write(f"IMG_SIZE: {IMG_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"PATIENCE: {PATIENCE}\n")
        f.write(f"BATCH: {BATCH}\n")
        f.write("Purpose: learn BAW-Sack/X-pattern evidence and reject YOLO false positives.\n")


def train():
    print("="*100)
    print("Training D2 crop verifier classifier")
    print("="*100)
    check_dataset()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    init_weight, init_reason = choose_initial_weight()

    run_name = f"D2_crop_verifier_yolov8m_cls_img{IMG_SIZE}_{timestamp()}"
    project_dir = dataset_runs_dir()
    planned = project_dir / run_name

    model = YOLO(str(init_weight))
    results = model.train(
        data=str(DATASET_DIR),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        seed=SEED,
        deterministic=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        plots=True,
        val=True,
        save=True,
        save_period=20,
        hsv_h=0.005, hsv_s=0.20, hsv_v=0.15,
        degrees=5.0, translate=0.04, scale=0.12,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        erasing=0.0,
        auto_augment=None,
        cos_lr=True,
    )

    run_dir = actual_save_dir(results, model, planned)
    best = run_dir / "weights" / "best.pt"
    write_info(run_dir, init_weight, init_reason)

    print(f"Best classifier: {best}")
    return best, run_dir


def validate(best, split):
    YOLO(str(best)).val(
        data=str(DATASET_DIR),
        split=split,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(dataset_runs_dir()),
        name=f"D2_crop_verifier_{split}_eval_img{IMG_SIZE}_{timestamp()}",
        plots=True,
    )


def main():
    best, run_dir = train()
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found: {best}")
    for split in ["train", "val", "test"]:
        validate(best, split)
    with (run_dir / "training_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item","value"])
        w.writerow(["module","D2 crop verifier classifier"])
        w.writerow(["dataset_dir",str(DATASET_DIR)])
        w.writerow(["best_model",str(best)])
        w.writerow(["run_dir",str(run_dir)])
    print("ALL DONE")


if __name__ == "__main__":
    main()
