#!/usr/bin/env python3
"""Print the D1 training environment and GPU status."""
from __future__ import annotations
import json
import sys
from d1_train_common import environment_report


def main() -> int:
    report = environment_report()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("ultralytics_error"):
        print("ERROR: Ultralytics is unavailable in this environment.", file=sys.stderr)
        return 1
    if not report.get("cuda_available"):
        print("WARNING: CUDA is not available; training will run on CPU unless the environment is corrected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
