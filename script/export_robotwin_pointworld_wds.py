"""Export RoboTwin compact PointWorld H5 clips to WebDataset shards.

This is a thin wrapper around PointWorld-data. It keeps the RoboTwin repo from
duplicating WDS conversion logic while documenting the exact command sequence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert RoboTwin compact PointWorld H5 files to WDS shards.")
    parser.add_argument("--input_dir", required=True, help="Directory containing compact episode*.hdf5 files.")
    parser.add_argument("--output_dir", required=True, help="Output WDS directory.")
    parser.add_argument("--pointworld_data_repo", default="/data/dex/PointWorld-data")
    parser.add_argument(
        "--pointworld_python",
        default=sys.executable,
        help="Python executable with PointWorld-data dependencies such as webdataset.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_percentage", type=float, default=0.1)
    parser.add_argument("--max_clips", type=int, default=-1)
    parser.add_argument("--num_mp_workers", type=int, default=1)
    parser.add_argument("--maxsize", type=float, default=1e9)
    parser.add_argument("--fast_integrity", action="store_true", help="Only check clip discovery before WDS export.")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    pointworld_data = Path(args.pointworld_data_repo).resolve()
    pointworld_python = str(Path(args.pointworld_python).expanduser().resolve())

    for script_name in ("data_integrity_check.py", "make_wds_manifest.py", "convert_wds.py"):
        script_path = pointworld_data / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Missing PointWorld-data script: {script_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    integrity_path = input_dir / "integrity_check.json"
    manifest_path = output_dir / "robotwin_manifest.json"

    integrity_cmd = [
        pointworld_python,
        "data_integrity_check.py",
        "--input_dir",
        str(input_dir),
        "--domain",
        "behavior",
        "--output_file",
        str(integrity_path),
        "--num_mp_workers",
        str(args.num_mp_workers),
    ]
    if args.fast_integrity:
        integrity_cmd.append("--fastmode")
    run(integrity_cmd, pointworld_data)

    run(
        [
            pointworld_python,
            "make_wds_manifest.py",
            "--input_dir",
            str(input_dir),
            "--domain",
            "robotwin",
            "--output_manifest",
            str(manifest_path),
            "--integrity_check_file",
            str(integrity_path),
            "--seed",
            str(args.seed),
            "--test_percentage",
            str(args.test_percentage),
            "--max_clips",
            str(args.max_clips),
        ],
        pointworld_data,
    )

    run(
        [
            pointworld_python,
            "convert_wds.py",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--domain",
            "robotwin",
            "--manifest",
            str(manifest_path),
            "--integrity_check_file",
            str(integrity_path),
            "--seed",
            str(args.seed),
            "--max_clips",
            str(args.max_clips),
            "--maxsize",
            str(args.maxsize),
        ],
        pointworld_data,
    )


if __name__ == "__main__":
    main()
