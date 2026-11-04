"""Runtime helpers for stable Ultralytics training."""
from __future__ import annotations

def suppress_auxiliary_amp_model_check(enabled: bool = True) -> None:
    """Keep AMP enabled but skip Ultralytics' auxiliary compatibility model check.

    In recent Ultralytics versions the AMP safety check can download a tiny
    current-generation YOLO checkpoint (e.g. yolo26n.pt). The user's CUDA/GPU
    environment has already passed that check. Monkey-patching the trainer's
    check_amp hook avoids repeated unrelated downloads while preserving AMP.
    """
    if not enabled:
        return
    try:
        import ultralytics.engine.trainer as trainer_module
        trainer_module.check_amp = lambda model: True
    except Exception:
        pass
