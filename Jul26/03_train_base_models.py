#!/usr/bin/env python3
"""Train the finalized three-stage Phase-4 architecture with robust initializer discovery."""
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fmt(path, threshold):
    return Path(str(path).replace("{T}", str(threshold))).expanduser().resolve()


def run(command, title, allow_fail=False):
    command = [str(value) for value in command]
    print("\n" + "=" * 88 + "\n" + title + "\n" + "=" * 88)
    print(" ".join(command))
    environment = dict(os.environ)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    result = subprocess.run(command, env=environment)
    if result.returncode and not allow_fail:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def read_manifest_checkpoint(final_models_root: Path, stage_key: str) -> Path | None:
    for manifest_name in ("model_manifest.json", "final_model_manifest.json"):
        manifest_path = final_models_root / manifest_name
        if not manifest_path.is_file():
            continue
        try:
            manifest = load(manifest_path)
        except Exception:
            continue
        models = manifest.get("models", {})
        # Supported manifests from both consolidation utilities.
        aliases = {
            "stage_a": ("stage_a", "StageA_AnyFM_Detector"),
            "stage_b": ("stage_b", "StageB_Binary_Verifier"),
            "stage_c": ("stage_c", "StageC_Exact_Classifier"),
        }[stage_key]
        for alias in aliases:
            value = models.get(alias)
            if isinstance(value, str):
                candidate = Path(value).expanduser()
                if candidate.is_file():
                    return candidate.resolve()
            if isinstance(value, dict):
                for field in ("copied_checkpoint", "checkpoint", "best_pt", "source_checkpoint"):
                    raw = value.get(field)
                    if raw and Path(raw).expanduser().is_file():
                        return Path(raw).expanduser().resolve()
    return None


def find_checkpoint(
    explicit: str,
    final_models_root: str,
    stage_key: str,
    stage_folder_name: str,
    search_root: str,
    candidate_folder_names: list[str],
) -> tuple[Path | None, str]:
    """Resolve a selected checkpoint in a strict, transparent priority order."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve(), "explicit checkpoint path"

    final_root = Path(final_models_root).expanduser() if final_models_root else None
    if final_root and final_root.is_dir():
        direct = final_root / stage_folder_name / "best.pt"
        if direct.is_file():
            return direct.resolve(), "PHASE4_FINAL_MODELS direct path"
        manifested = read_manifest_checkpoint(final_root, stage_key)
        if manifested is not None:
            return manifested, "PHASE4_FINAL_MODELS manifest"

    root = Path(search_root).expanduser()
    if root.is_dir():
        # Also discover a consolidated folder nested under the search root.
        direct_final = root / "PHASE4_FINAL_MODELS" / stage_folder_name / "best.pt"
        if direct_final.is_file():
            return direct_final.resolve(), "search_root/PHASE4_FINAL_MODELS"

        # Legacy candidate run folders.
        for folder in candidate_folder_names:
            direct = root / folder / "weights" / "best.pt"
            if direct.is_file():
                return direct.resolve(), f"legacy candidate folder {folder}"

        # Recursive fallback, deterministic and stage-aware.
        expected_suffixes = [
            Path("PHASE4_FINAL_MODELS") / stage_folder_name / "best.pt",
            *[Path(folder) / "weights" / "best.pt" for folder in candidate_folder_names],
        ]
        for candidate in sorted(root.rglob("best.pt"), key=lambda p: str(p).casefold()):
            for suffix in expected_suffixes:
                try:
                    if candidate.parts[-len(suffix.parts):] == suffix.parts:
                        return candidate.resolve(), "recursive fallback"
                except Exception:
                    pass

    return None, "not found"


def resolve_initializers(config: dict) -> tuple[dict, dict]:
    training = config["training"]
    pretrained = config["pretrained"]
    force = bool(pretrained.get("force_scratch", False))
    generic = bool(pretrained.get("use_generic_pretrained_when_selected_missing", True))
    search_root = pretrained.get("search_root", "")
    final_root = pretrained.get("final_models_root", "")

    stage_specs = {
        "stage_a": (
            training["stage_a"],
            "StageA_AnyFM_Detector",
            [training["stage_a"]["selected_folder_name"]],
        ),
        "stage_b": (
            training["stage_b"],
            "StageB_Binary_Verifier",
            [training["stage_b"]["selected_folder_name"]],
        ),
        "stage_c": (
            training["stage_c"],
            "StageC_Exact_Classifier",
            [
                training["stage_c"]["selected_folder_name"],
                *training["stage_c"].get("fallback_selected_folder_names", []),
            ],
        ),
    }

    selected = {}
    diagnostics = {}
    for stage_key, (settings, stage_folder, legacy_names) in stage_specs.items():
        if force:
            checkpoint, source = None, "force_scratch=true"
        else:
            checkpoint, source = find_checkpoint(
                pretrained.get(f"{stage_key}_checkpoint", ""),
                final_root,
                stage_key,
                stage_folder,
                search_root,
                legacy_names,
            )
        selected[stage_key] = checkpoint
        initializer = checkpoint or (
            settings["generic_pretrained"]
            if generic and not force
            else settings["scratch_architecture"]
        )
        diagnostics[stage_key] = {
            "selected_checkpoint": str(checkpoint) if checkpoint else None,
            "source": source,
            "training_initializer": str(initializer),
            "fallback_if_missing": (
                settings["generic_pretrained"]
                if generic and not force
                else settings["scratch_architecture"]
            ),
        }
    return selected, diagnostics


def print_initializer_report(config: dict, diagnostics: dict) -> None:
    print("\n" + "=" * 88)
    print("PRETRAINED MODEL DISCOVERY REPORT")
    print("=" * 88)
    print(f"Configured final_models_root: {config['pretrained'].get('final_models_root', '')}")
    print(f"Configured search_root:       {config['pretrained'].get('search_root', '')}")
    for stage_key in ("stage_a", "stage_b", "stage_c"):
        item = diagnostics[stage_key]
        print(f"\n{stage_key.upper()}")
        print(f"  selected checkpoint: {item['selected_checkpoint']}")
        print(f"  discovery source:    {item['source']}")
        print(f"  training initializer:{item['training_initializer']}")
    selected_count = sum(item["selected_checkpoint"] is not None for item in diagnostics.values())
    print(f"\nSelected Phase-4 checkpoints found: {selected_count}/3")
    if selected_count < 3:
        print("Missing stages will use the configured generic pretrained model or scratch architecture.")


def make_empty_hard_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        csv.DictWriter(
            stream,
            fieldnames=[
                "image", "relative_image", "instances", "hard_objects",
                "image_hardness", "classes", "context",
            ],
        ).writeheader()


def train_detector(data, project, name, model, settings, config, overwrite=True):
    command = [
        sys.executable, ROOT / "train_anyfm_detector.py",
        "--data", data, "--project", project, "--name", name,
        "--model", model, "--mosaic", settings["mosaic"],
        "--imgsz", settings["imgsz"], "--epochs", settings["epochs"],
        "--patience", settings["patience"], "--batch", settings["batch"],
        "--device", config["device"], "--workers", config["workers"],
        "--seed", config["seed"], "--optimizer", settings["optimizer"],
        "--lr0", settings["lr0"], "--lrf", settings["lrf"],
        "--momentum", settings["momentum"],
        "--weight-decay", settings["weight_decay"],
        "--warmup-epochs", settings["warmup_epochs"],
        "--close-mosaic", settings["close_mosaic"],
        "--amp", "true" if config["training"].get("amp", True) else "false",
    ]
    if config["training"].get("skip_ultralytics_amp_check", True):
        command.append("--skip-amp-check")
    if overwrite:
        command.append("--overwrite")
    rc = run(
        command,
        f"TRAIN {name}",
        allow_fail=str(model).endswith(".pt") and not Path(str(model)).is_file(),
    )
    if rc:
        command[command.index("--model") + 1] = settings["scratch_architecture"]
        run(command, f"PRETRAINED UNAVAILABLE — TRAIN {name} FROM SCRATCH")
    return Path(project) / name / "weights" / "best.pt"


def train_classifier(data, project, name, model, role, settings, config, epochs=None, patience=None, overwrite=True):
    command = [
        sys.executable, ROOT / "train_phase4_classifier.py",
        "--data", data, "--project", project, "--name", name,
        "--model", model, "--role", role, "--imgsz", settings["imgsz"],
        "--epochs", epochs or settings["epochs"],
        "--patience", patience or settings["patience"],
        "--batch", settings["batch"], "--device", config["device"],
        "--workers", config["workers"], "--seed", config["seed"],
        "--optimizer", settings["optimizer"], "--lr0", settings["lr0"],
        "--lrf", settings["lrf"], "--weight-decay", settings["weight_decay"],
        "--warmup-epochs", settings["warmup_epochs"],
        "--dropout", settings["dropout"],
        "--amp", "true" if config["training"].get("amp", True) else "false",
    ]
    if config["training"].get("skip_ultralytics_amp_check", True):
        command.append("--skip-amp-check")
    if overwrite:
        command.append("--overwrite")
    rc = run(
        command,
        f"TRAIN {name}",
        allow_fail=str(model).endswith(".pt") and not Path(str(model)).is_file(),
    )
    if rc:
        command[command.index("--model") + 1] = settings["scratch_architecture"]
        run(command, f"PRETRAINED UNAVAILABLE — TRAIN {name} FROM SCRATCH")
    return Path(project) / name / "weights" / "best.pt"


def copy_final(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target / "best.pt")
    last = source.with_name("last.pt")
    if last.is_file():
        shutil.copy2(last, target / "last.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--check-initializers",
        action="store_true",
        help="Resolve and report all pretrained models without starting training.",
    )
    args = parser.parse_args()
    config = load(args.config)
    threshold = int(config["threshold"])

    selected, diagnostics = resolve_initializers(config)
    print_initializer_report(config, diagnostics)
    if args.check_initializers:
        return 0 if all(selected.values()) else 2

    dataset = fmt(config["dataset_output"], threshold) / "detection"
    runs = fmt(config["runs_root"], threshold)
    assets = runs / "TRAINING_ASSETS"
    project = runs / "TRAINING_RUNS"
    final = runs / "BASE_FINAL_MODELS"
    runs.mkdir(parents=True, exist_ok=True)
    if not (dataset / "dataset.yaml").is_file():
        raise FileNotFoundError(f"Create dataset first: {dataset}")

    existing_manifest = final / "base_model_manifest.json"
    if (
        existing_manifest.is_file()
        and config["training"].get("reuse_completed_base_runs", True)
        and not args.overwrite
    ):
        existing = load(existing_manifest)
        missing = [
            path for path in existing.get("models", {}).values()
            if not Path(path).is_file()
        ]
        if not missing:
            print("Completed BASE_FINAL_MODELS already exist; reusing them. Add --overwrite to retrain.")
            print(json.dumps(existing.get("models", {}), indent=2))
            return 0

    training = config["training"]
    A, B, C = training["stage_a"], training["stage_b"], training["stage_c"]
    mine = training["crop_mining"]
    selected_a, selected_b, selected_c = selected["stage_a"], selected["stage_b"], selected["stage_c"]
    initializers = {
        key: value["training_initializer"] for key, value in diagnostics.items()
    }

    empty = assets / "empty_hard_positive_images.csv"
    make_empty_hard_csv(empty)
    base_view = assets / "StageA_Base_View"
    run([
        sys.executable, ROOT / "build_anyfm_replay_view.py",
        "--data", dataset, "--hard-images", empty, "--output", base_view,
        "--replay-fraction", 0.0, "--negative-ratio", A["negative_ratio"],
        "--seed", config["seed"], "--copy-mode", config["copy_mode"], "--overwrite",
    ], "BUILD STAGE-A BASE VIEW")

    audit_model = selected_a
    if audit_model is None:
        audit_model = train_detector(
            base_view, project, "StageA_Initial_NoReplay", initializers["stage_a"],
            {**A, "mosaic": 0.10}, config, True,
        )

    audit = assets / "StageA_Hard_Positive_Audit"
    run([
        sys.executable, ROOT / "audit_hard_positives.py",
        "--detector", audit_model, "--data", dataset, "--output", audit,
        "--split", "train", "--det-conf-min", A["audit_confidence_floor"],
        "--low-conf", A["audit_low_confidence"], "--poor-iou", A["audit_poor_iou"],
        "--imgsz", A["imgsz"], "--batch", A["batch"], "--device", config["device"],
        "--nms-iou", A["nms_iou"],
    ], "AUDIT HARD POSITIVES")

    final_a_view = assets / "StageA_Final_Replay_View"
    run([
        sys.executable, ROOT / "build_anyfm_replay_view.py",
        "--data", dataset, "--hard-images", audit / "hard_positive_images.csv",
        "--output", final_a_view,
        "--replay-fraction", A["hard_positive_replay_fraction"],
        "--negative-ratio", A["negative_ratio"], "--seed", config["seed"],
        "--copy-mode", config["copy_mode"], "--overwrite",
    ], "BUILD FINAL STAGE-A VIEW")
    final_a = train_detector(
        final_a_view, project, "StageA_AnyFM_Detector",
        str(selected_a or audit_model), A, config, True,
    )

    initial_exact_view = assets / "StageC_Initial_GT_Crops"
    run([
        sys.executable, ROOT / "build_initial_exact_view.py",
        "--data", dataset, "--output", initial_exact_view,
        "--crop-margin", mine["crop_margin"], "--balance-target", 0.90,
        "--seed", config["seed"], "--copy-mode", config["copy_mode"], "--overwrite",
    ], "BUILD INITIAL EXACT-CLASS VIEW")
    initial_c = train_classifier(
        initial_exact_view, project, "StageC_Initial_Exact_Classifier",
        initializers["stage_c"], "exact", C, config,
        C["initial_epochs"], C["initial_patience"], True,
    )

    pool = assets / "StageBC_Crop_Pool"
    run([
        sys.executable, ROOT / "build_stagebc_crop_pool.py",
        "--detector", final_a, "--exact-initializer", initial_c,
        "--data", dataset, "--output", pool,
        "--det-conf-min", mine["detector_confidence_floor"],
        "--nms-iou", mine["nms_iou"], "--imgsz", A["imgsz"],
        "--batch", A["batch"], "--device", config["device"],
        "--crop-margin", mine["crop_margin"],
        "--max-gt-iou", mine["maximum_iou_with_any_gt"],
        "--max-iop", mine["maximum_intersection_over_proposal"],
        "--max-background-per-image", mine["maximum_background_crops_per_image"],
        "--low-exact-conf", mine["exact_low_confidence"],
        "--low-exact-margin", mine["exact_low_margin"], "--overwrite",
    ], "BUILD STAGE-B/C HARD CROP POOL")

    views = assets / "StageBC_Final_Views"
    run([
        sys.executable, ROOT / "build_classifier_views.py",
        "--pool", pool, "--output", views,
        "--background-ratios", B["background_ratio"],
        "--hard-replay-fractions", C["hard_replay_fraction"],
        "--negative-image-fraction", B["preferred_negative_image_fraction"],
        "--seed", config["seed"], "--copy-mode", config["copy_mode"], "--overwrite",
    ], "BUILD FINAL STAGE-B AND STAGE-C VIEWS")

    btag = f"BG_{B['background_ratio']:.2f}".replace(".", "p")
    ctag = f"Hard_{C['hard_replay_fraction']:.2f}".replace(".", "p")
    final_b = train_classifier(
        views / "binary_views" / btag, project, "StageB_Binary_Verifier",
        initializers["stage_b"], "verifier", B, config, overwrite=True,
    )
    final_c = train_classifier(
        views / "exact_views" / ctag, project, "StageC_Exact_Classifier",
        str(initial_c), "exact", C, config, overwrite=True,
    )

    if final.exists():
        shutil.rmtree(final)
    copy_final(final_a, final / "StageA_AnyFM_Detector")
    copy_final(final_b, final / "StageB_Binary_Verifier")
    copy_final(final_c, final / "StageC_Exact_Classifier")
    manifest = {
        "threshold": threshold,
        "dataset": str(dataset),
        "runs_root": str(runs),
        "pretrained_discovery": diagnostics,
        "initializers": initializers,
        "models": {
            "stage_a": str((final / "StageA_AnyFM_Detector" / "best.pt").resolve()),
            "stage_b": str((final / "StageB_Binary_Verifier" / "best.pt").resolve()),
            "stage_c": str((final / "StageC_Exact_Classifier" / "best.pt").resolve()),
        },
        "default_operating_point": config["default_operating_point"],
        "operating_points": config["operating_points"],
        "model_role": "base models trained from the project target dataset",
        "scientific_status": (
            "Phase-4/4B engineering architecture; Phase-4B failed locked master validation. "
            "Phase 3 remains the validated model."
        ),
    }
    (final / "base_model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nFINAL MODELS:\n" + json.dumps(manifest["models"], indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
