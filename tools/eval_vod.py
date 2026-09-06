#!/usr/bin/env python3
"""Evaluate an OpenPCDet result.pkl with the View-of-Delft devkit."""

import argparse
import json
import pickle
import shutil
import sys
import types
from pathlib import Path


CLASS_NAMES = ("Car", "Pedestrian", "Cyclist")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-pkl", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--vod-devkit", default=Path("/home/car/VoD-evaluation"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-thresh", default=-1.0, type=float)
    return parser.parse_args()


def export_kitti_predictions(result_pkl: Path, output_dir: Path):
    with result_pkl.open("rb") as f:
        predictions = pickle.load(f)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    for prediction in predictions:
        frame_id = str(prediction["frame_id"])
        with (output_dir / f"{frame_id}.txt").open("w") as f:
            for idx, name in enumerate(prediction["name"]):
                bbox = prediction["bbox"][idx]
                dims = prediction["dimensions"][idx]  # l, h, w
                loc = prediction["location"][idx]
                print(
                    f"{name} -1 -1 {prediction['alpha'][idx]:.4f} "
                    f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f} "
                    f"{dims[1]:.4f} {dims[2]:.4f} {dims[0]:.4f} "
                    f"{loc[0]:.4f} {loc[1]:.4f} {loc[2]:.4f} "
                    f"{prediction['rotation_y'][idx]:.4f} {prediction['score'][idx]:.4f}",
                    file=f,
                )

    return len(predictions)


def main():
    args = parse_args()
    for path in (args.result_pkl, args.label_dir, args.vod_devkit):
        if not path.exists():
            raise FileNotFoundError(path)

    frame_count = export_kitti_predictions(args.result_pkl, args.output_dir)

    # The devkit package imports optional visualization dependencies (k3d,
    # matplotlib) from vod/__init__.py. Evaluation itself does not need them,
    # so register the package path without executing that top-level module.
    vod_package = types.ModuleType("vod")
    vod_package.__path__ = [str(args.vod_devkit / "vod")]
    sys.modules["vod"] = vod_package
    from vod.common.file_handling import get_frame_list_from_folder
    vod_package.get_frame_list_from_folder = get_frame_list_from_folder
    from vod.evaluation import Evaluation

    results = Evaluation(test_annotation_file=str(args.label_dir)).evaluate(
        result_path=str(args.output_dir),
        current_class=[0, 1, 2],
        score_thresh=args.score_thresh,
    )

    summary = {"frames": frame_count}
    for area in ("entire_area", "roi"):
        class_ap = {
            name: float(results[area][f"{name}_3d_all"])
            for name in CLASS_NAMES
        }
        summary[area] = {
            **class_ap,
            "mAP": sum(class_ap.values()) / len(class_ap),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
