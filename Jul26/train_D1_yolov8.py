#!/usr/bin/env python3
"""Train YOLOv8 image classification on D1."""
from __future__ import annotations
import argparse
import sys
from d1_train_common import add_training_arguments, train_classifier


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8 classification on D1 Wagon/Container dataset.")
    add_training_arguments(parser, default_model="yolov8m-cls.pt", default_name="yolov8m_D1_640_noaug")
    args = parser.parse_args()
    try:
        train_classifier(args, family="YOLOv8")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
