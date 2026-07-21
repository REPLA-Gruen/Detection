from pathlib import Path
from ultralytics import YOLO
import time
import csv

# ============================================================
# YOLOv8 TRAINING — D3
# PATH-FIXED VERSION
# ============================================================
#
# Fix:
#   The script no longer creates the run folder before model.train().
#   After training, it gets the real Ultralytics save_dir and uses that.
#   This prevents the "folder name + 2" best.pt path error.
# ============================================================

DATASET_KEY = "D3"

DATASET_YAML = Path(
    r"C:\Hadi\Foreign_Material\Data\Ours\Jun-26_Focused_Datasets_With_Negatives_Optimized_80_10_10\D3_Plastiksack_binary_with_negatives\data.yaml"
)

RUNS_ROOT = Path(
    r"C:\Hadi\Foreign_Material\Runs\Focused_Datasets_YOLOv8_NoAug_Img1536_Separate"
)

DATASET_RUN_NAME = "D3_Plastiksack_binary_with_negatives"

BASE_MODEL = "yolov8m.pt"

IMG_SIZE = 1536
EPOCHS = 200
PATIENCE = 50
BATCH = 1
WORKERS = 4
DEVICE = 0
SEED = 42

# First baseline should be False.
# Later improvement runs can set this to True.
USE_PREVIOUS_BEST_FROM_THIS_DATASET = False


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def dataset_runs_dir():
    return RUNS_ROOT / DATASET_RUN_NAME


def find_previous_best():
    root = dataset_runs_dir()
    if not root.exists():
        return None

    candidates = []
    for p in root.rglob("weights/best.pt"):
        try:
            candidates.append((p.stat().st_mtime, p))
        except OSError:
            pass

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def choose_initial_weight():
    if USE_PREVIOUS_BEST_FROM_THIS_DATASET:
        previous_best = find_previous_best()
        if previous_best is not None:
            return previous_best, "previous best.pt from this same dataset run folder"

    return BASE_MODEL, "base YOLOv8 pretrained model"


def get_actual_save_dir(results, model, fallback_dir):
    """
    Ultralytics may change the run folder name if a folder already exists.
    This function always returns the actual save directory.
    """
    save_dir = getattr(results, "save_dir", None)

    if save_dir is None and hasattr(model, "trainer"):
        save_dir = getattr(model.trainer, "save_dir", None)

    if save_dir is None:
        save_dir = fallback_dir

    return Path(save_dir)


def write_run_info(run_dir, init_weight, init_reason):
    run_dir.mkdir(parents=True, exist_ok=True)
    info_path = run_dir / "training_info.txt"

    with info_path.open("w", encoding="utf-8") as f:
        f.write(f"Dataset key: {DATASET_KEY}\n")
        f.write(f"Dataset name: {DATASET_RUN_NAME}\n")
        f.write(f"Dataset YAML: {DATASET_YAML}\n")
        f.write("Task: detection\n")
        f.write(f"Initial weight: {init_weight}\n")
        f.write(f"Initial reason: {init_reason}\n")
        f.write(f"Base model: {BASE_MODEL}\n")
        f.write(f"IMG_SIZE: {IMG_SIZE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"PATIENCE: {PATIENCE}\n")
        f.write(f"BATCH: {BATCH}\n")
        f.write(f"WORKERS: {WORKERS}\n")
        f.write(f"DEVICE: {DEVICE}\n")
        f.write(f"SEED: {SEED}\n")
        f.write("Augmentation: disabled\n")
        f.write(f"Actual run folder: {run_dir}\n")


def train():
    print("=" * 100)
    print("YOLOv8 training — D3: D3_Plastiksack_binary_with_negatives")
    print("=" * 100)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET_YAML}")

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    init_weight, init_reason = choose_initial_weight()

    print("Initialization:")
    print(f"  {init_weight}")
    print(f"Reason:")
    print(f"  {init_reason}")
    print()

    run_name = f"{DATASET_KEY}_yolov8m_detect_noaug_img{IMG_SIZE}_{timestamp()}"
    project_dir = dataset_runs_dir()
    planned_run_dir = project_dir / run_name

    # IMPORTANT:
    # Do NOT create planned_run_dir before model.train().
    # Ultralytics creates the real folder itself.

    model = YOLO(str(init_weight))

    results = model.train(
        data=str(DATASET_YAML),
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
        save_period=25,

        # No augmentation
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,
        auto_augment=None,
        close_mosaic=0,
        cos_lr=True,
    )

    run_dir = get_actual_save_dir(results, model, planned_run_dir)
    best_model = run_dir / "weights" / "best.pt"

    write_run_info(run_dir, init_weight, init_reason)

    print()
    print("Training finished.")
    print(f"Actual run folder:")
    print(f"  {run_dir}")
    print(f"Best model:")
    print(f"  {best_model}")
    print()

    return best_model, run_dir


def validate_test(best_model):
    print("=" * 100)
    print("Test validation")
    print("=" * 100)

    if not best_model.exists():
        raise FileNotFoundError(f"best.pt not found: {best_model}")

    model = YOLO(str(best_model))

    model.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(dataset_runs_dir()),
        name=f"{DATASET_KEY}_test_eval_noaug_img{IMG_SIZE}_{timestamp()}",
        plots=True,
    )


def main():
    best_model, run_dir = train()
    validate_test(best_model)

    summary_csv = run_dir / "training_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item", "value"])
        writer.writerow(["dataset_key", DATASET_KEY])
        writer.writerow(["dataset_name", DATASET_RUN_NAME])
        writer.writerow(["dataset_yaml", str(DATASET_YAML)])
        writer.writerow(["task", "detection"])
        writer.writerow(["best_model", str(best_model)])
        writer.writerow(["run_dir", str(run_dir)])
        writer.writerow(["img_size", IMG_SIZE])
        writer.writerow(["augmentation", "disabled"])

    print("=" * 100)
    print("ALL DONE")
    print("=" * 100)
    print(f"Summary CSV:")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
