from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("No command provided after --")

    run_root = Path(args.run_root)
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{args.exp_id}_train_console.log"
    json_path = log_dir / f"{args.exp_id}_timing.json"

    start_wall = datetime.now()
    start = time.perf_counter()

    print("=" * 80)
    print(f"Experiment: {args.exp_id}")
    print(f"Start: {start_wall.isoformat(timespec='seconds')}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 80)

    return_code = None
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("=" * 80 + "\n")
        log.write(f"Experiment: {args.exp_id}\n")
        log.write(f"Start: {start_wall.isoformat(timespec='seconds')}\n")
        log.write("Command:\n")
        log.write(" ".join(cmd) + "\n")
        log.write("=" * 80 + "\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )

        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        return_code = proc.wait()

    end = time.perf_counter()
    end_wall = datetime.now()
    elapsed = end - start

    summary = {
        "experiment": args.exp_id,
        "start_time": start_wall.isoformat(timespec="seconds"),
        "end_time": end_wall.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "elapsed_hours": elapsed / 3600.0,
        "return_code": return_code,
        "log_path": str(log_path),
        "command": cmd,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"Experiment finished: {args.exp_id}")
    print(f"Elapsed: {elapsed/60.0:.2f} minutes")
    print(f"Return code: {return_code}")
    print(f"Timing JSON: {json_path}")
    print("=" * 80)

    if return_code != 0:
        sys.exit(return_code)

if __name__ == "__main__":
    main()
