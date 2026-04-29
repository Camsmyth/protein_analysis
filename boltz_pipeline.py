#!/usr/bin/env python3
"""
Boltz2 protein structure prediction pipeline.

Reads protein sequences from a CSV or Excel file and runs Boltz2 structure
prediction for each sequence. Output structure files are written flat into
the output directory as {Sample}-1.cif, {Sample}-2.cif, etc.

Usage:
    python boltz_pipeline.py input.csv
    python boltz_pipeline.py input.xlsx --names ID --sequences Seq
    python boltz_pipeline.py input.csv --num-models 5 --use-msa-server
    python boltz_pipeline.py input.csv --accelerator cpu --output results/
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from monitor import PowerMetricsMonitor, launch_asitop

BOLTZ_BIN = Path(sys.executable).parent / "boltz"


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------

def write_boltz_yaml(sequence: str, path: Path) -> None:
    path.write_text(
        f"sequences:\n"
        f"  - protein:\n"
        f"      id: A\n"
        f"      sequence: \"{sequence}\"\n"
    )


# ---------------------------------------------------------------------------
# Boltz runner
# ---------------------------------------------------------------------------

def run_boltz(
    input_dir: Path,
    out_dir: Path,
    accelerator: str,
    num_models: int,
    no_msa_server: bool,
) -> int:
    cmd = [
        str(BOLTZ_BIN), "predict", str(input_dir),
        "--out_dir", str(out_dir),
        "--accelerator", accelerator,
        "--diffusion_samples", str(num_models),
        "--model", "boltz2",
    ]
    if not no_msa_server:
        cmd.append("--use_msa_server")

    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Output post-processing
# ---------------------------------------------------------------------------

def collect_structures(boltz_out: Path, final_out: Path, output_format: str) -> int:
    """
    Boltz writes: predictions/{name}/{name}_model_{N}.cif
    Rename and flatten to:  {name}-{N+1}.cif
    Returns the number of files moved.
    """
    ext = "cif" if output_format == "mmcif" else "pdb"
    moved = 0

    for src in sorted(boltz_out.glob(f"*/predictions/*/*.{ext}")):
        stem = src.stem  # e.g. "MySample_model_0"
        if "_model_" not in stem:
            continue
        name, rank_str = stem.rsplit("_model_", 1)
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        dst = final_out / f"{name}-{rank + 1}.{ext}"
        shutil.move(str(src), str(dst))
        print(f"  {src.name}  ->  {dst.name}")
        moved += 1

    return moved


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_input(filepath: str, names_col: str, sequences_col: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"Error: file not found — {filepath}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        sys.exit(f"Error: unsupported format '{suffix}'. Use .csv, .xlsx, or .xls.")

    missing = [c for c in [names_col, sequences_col] if c not in df.columns]
    if missing:
        sys.exit(
            f"Error: column(s) not found: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )
    return df


def clean_sequence(raw: object) -> str:
    return str(raw).strip().upper().replace(" ", "").replace("\n", "")


def make_safe_name(raw: object) -> str:
    return str(raw).strip().replace(" ", "_").replace("/", "_")


def default_output_dir(input_path: str) -> Path:
    p = Path(input_path)
    return p.parent / "boltz_predictions"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Boltz2 structure prediction pipeline from CSV or Excel input."
    )
    parser.add_argument("input", help="Input CSV or Excel file")
    parser.add_argument(
        "--names",
        default="Sample",
        metavar="COLUMN",
        help="Column containing sample/molecule names (default: Sample)",
    )
    parser.add_argument(
        "--sequences",
        default="Sequence",
        metavar="COLUMN",
        help="Column containing protein sequences (default: Sequence)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for predictions. Defaults to boltz_predictions/ "
            "in the same directory as the input file."
        ),
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        choices=["gpu", "cpu", "tpu"],
        help="Hardware accelerator (default: gpu)",
    )
    parser.add_argument(
        "--num-models",
        type=int,
        default=1,
        metavar="N",
        help="Number of structure models to generate per sequence (default: 1)",
    )
    parser.add_argument(
        "--output-format",
        default="mmcif",
        choices=["mmcif", "pdb"],
        help="Structure output format (default: mmcif)",
    )
    parser.add_argument(
        "--no-msa-server",
        action="store_true",
        help="Disable MSA server lookup (only use if providing pre-computed MSAs)",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable asitop and powermetrics monitoring during the run",
    )
    args = parser.parse_args()

    df = load_input(args.input, args.names, args.sequences)
    total = len(df)
    print(f"Loaded {total} sequences from '{args.input}'")
    print(f"  Names column    : '{args.names}'")
    print(f"  Sequences column: '{args.sequences}'\n")

    out_dir = Path(args.output) if args.output else default_output_dir(args.input)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    # --- monitoring setup ---
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    monitor = None
    if not args.no_monitor:
        launch_asitop()
        monitor = PowerMetricsMonitor(
            log_path=out_dir / f"powermetrics_{run_ts}.log"
        )
        monitor.start()

    # Boltz writes into a persistent raw/ subdir — structures survive any post-processing crash
    boltz_out = out_dir / "raw"
    boltz_out.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as yaml_tmp:
        yaml_dir = Path(yaml_tmp)
        skipped = 0

        for _, row in df.iterrows():
            seq = clean_sequence(row[args.sequences])
            name = make_safe_name(row[args.names])

            if not seq:
                print(f"  Skipping '{name}' — empty sequence.", file=sys.stderr)
                skipped += 1
                continue

            write_boltz_yaml(seq, yaml_dir / f"{name}.yaml")

        queued = total - skipped
        print(f"Queued {queued}/{total} sequences for prediction.\n")

        if queued == 0:
            if monitor:
                monitor.stop()
            sys.exit("No valid sequences to predict.")

        returncode = run_boltz(
            input_dir=yaml_dir,
            out_dir=boltz_out,
            accelerator=args.accelerator,
            num_models=args.num_models,
            no_msa_server=args.no_msa_server,
        )

    if monitor:
        monitor.stop()
        try:
            monitor.plot(out_dir / f"usage_{run_ts}.png")
        except Exception as e:
            print(f"Warning: usage plot failed — {e}", file=sys.stderr)

    if returncode != 0:
        print(f"\nBoltz exited with code {returncode}.", file=sys.stderr)
        sys.exit(returncode)

    print("\nCollecting output structures...")
    moved = collect_structures(boltz_out, out_dir, args.output_format)

    if moved:
        print(f"\nDone. {moved} structure file(s) saved to: {out_dir}")
    else:
        print("\nWarning: no structure files were found in Boltz output.", file=sys.stderr)


if __name__ == "__main__":
    main()
