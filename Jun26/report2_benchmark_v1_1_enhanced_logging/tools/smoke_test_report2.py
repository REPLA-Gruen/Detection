from pathlib import Path
import os, sys, json
from PIL import Image
import yaml

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def envp(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return Path(v)

def count_images(path):
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in EXTS)

def inspect_sizes(path, max_n=200):
    sizes = []
    if not path.exists():
        return {}
    for p in path.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            with Image.open(p) as im:
                sizes.append((im.width, im.height))
        except Exception:
            pass
        if len(sizes) >= max_n:
            break
    if not sizes:
        return {}
    ws = [x[0] for x in sizes]
    hs = [x[1] for x in sizes]
    return {"sampled_images": len(sizes), "min_width": min(ws), "max_width": max(ws), "min_height": min(hs), "max_height": max(hs), "max_dimension_sample": max(max(ws), max(hs))}

def main():
    run_root = envp("RUN_ROOT")
    run_root.mkdir(parents=True, exist_ok=True)
    data_a, data_b1, data_b2 = envp("DATA_A"), envp("DATA_B1"), envp("DATA_B2")
    data_a_yaml, data_b2_yaml = envp("DATA_A_YAML"), envp("DATA_B2_YAML")

    checks = []
    for label, path in [("DATA_A", data_a), ("DATA_B1", data_b1), ("DATA_B2", data_b2)]:
        checks.append({"check": f"{label}_exists", "ok": path.exists(), "value": str(path)})
        checks.append({"check": f"{label}_image_count", "ok": True, "value": count_images(path)})

    for label, ypath in [("DATA_A_YAML", data_a_yaml), ("DATA_B2_YAML", data_b2_yaml)]:
        ok = ypath.exists()
        checks.append({"check": f"{label}_exists", "ok": ok, "value": str(ypath)})
        if ok:
            cfg = yaml.safe_load(ypath.read_text(encoding="utf-8"))
            checks.append({"check": f"{label}_nc", "ok": True, "value": cfg.get("nc")})
            checks.append({"check": f"{label}_names", "ok": True, "value": cfg.get("names")})

    for split in ["train", "val", "test"]:
        for cls in ["Wagon", "Container"]:
            p = data_b1 / split / cls
            checks.append({"check": f"B1_{split}_{cls}", "ok": p.exists(), "value": count_images(p) if p.exists() else 0})

    try:
        import ultralytics
        checks.append({"check": "ultralytics_import", "ok": True, "value": getattr(ultralytics, "__version__", "unknown")})
    except Exception as e:
        checks.append({"check": "ultralytics_import", "ok": False, "value": str(e)})

    try:
        import torch
        checks.append({"check": "torch_cuda_available", "ok": bool(torch.cuda.is_available()), "value": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"})
    except Exception as e:
        checks.append({"check": "torch_cuda_available", "ok": False, "value": str(e)})

    out = {"checks": checks, "image_size_sample": {
        "DATA_A_train": inspect_sizes(data_a / "images" / "train"),
        "DATA_B2_train": inspect_sizes(data_b2 / "images" / "train"),
        "DATA_B1_train": inspect_sizes(data_b1 / "train"),
        "IMG_SIZE_config": os.environ.get("IMG_SIZE"),
    }}
    (run_root / "smoke_test_report2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if any(c.get("ok") is False for c in checks):
        sys.exit(1)

if __name__ == "__main__":
    main()
