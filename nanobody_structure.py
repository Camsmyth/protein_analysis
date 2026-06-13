#!/usr/bin/env python3
"""
NanobodyBuilder2 structure prediction from FASTA.

Runs ImmuneBuilder's NanoBodyBuilder2 on each sequence in a FASTA file,
producing 4 OpenMM-refined model structures per sequence. The 4 models come
from independently pre-trained networks and represent genuine uncertainty over
CDR3 loop geometry. All 4 are refined equally so they can be used directly as
docking templates or for visual inspection.

Output per sequence (in --output/{name}/):
    {name}_model_1.pdb  …  {name}_model_4.pdb   — refined conformers (rank 1=best)
    {name}_best.pdb                              — copy of rank-1 model

Usage:
    python nanobody_structure.py sequences.fasta
    python nanobody_structure.py sequences.fasta --output structures/
    python nanobody_structure.py sequences.fasta --no-refine
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    """Return list of (name, sequence) from a FASTA file."""
    entries: list[tuple[str, str]] = []
    name = None
    seqparts: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                entries.append((name, "".join(seqparts)))
            name = line[1:].split()[0]
            seqparts = []
        else:
            seqparts.append(line.upper())
    if name is not None:
        entries.append((name, "".join(seqparts)))
    return entries


def make_safe_name(raw: str) -> str:
    return raw.strip().replace(" ", "_").replace("/", "_").replace("|", "_")


def predict_one(
    name: str,
    sequence: str,
    out_dir: Path,
    refine_structures: bool,
    builder,
    refine_fn,
) -> list[Path]:
    """Run NanoBodyBuilder2 on one sequence. Returns list of saved PDB paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Predicting {name} ({len(sequence)} aa)…")

    try:
        nanobody = builder.predict({"H": sequence})
    except Exception as e:
        print(f"  ERROR: prediction failed for '{name}': {e}", file=sys.stderr)
        return []

    saved: list[Path] = []
    for rank, model_idx in enumerate(nanobody.ranking):
        pdb_path = out_dir / f"{name}_model_{rank + 1}.pdb"
        nanobody.save_single_unrefined(str(pdb_path), index=model_idx)

        if refine_structures:
            success = refine_fn(str(pdb_path), str(pdb_path))
            status = "refined ✓" if success else "refinement failed, kept unrefined"
            print(f"    model {rank + 1}: {status}")
        else:
            print(f"    model {rank + 1}: saved (no refinement)")

        saved.append(pdb_path)

    if saved:
        best = out_dir / f"{name}_best.pdb"
        shutil.copy2(saved[0], best)
        print(f"  Best model → {best.name}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NanobodyBuilder2 on all sequences in a FASTA file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("fasta", help="Input FASTA file containing VHH sequences.")
    parser.add_argument(
        "--output", default=None, metavar="DIR",
        help="Output directory. Default: nb2_structures/ next to the input file.",
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="Skip OpenMM refinement (faster, structures may have minor geometry issues).",
    )
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        sys.exit(f"Error: file not found — {args.fasta}")

    entries = parse_fasta(fasta_path)
    if not entries:
        sys.exit("Error: no sequences found in FASTA file.")

    out_root = Path(args.output) if args.output else fasta_path.parent / "nb2_structures"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(entries)} sequence(s) in {fasta_path.name}")
    print(f"Output: {out_root}")
    print(f"Refinement: {'off' if args.no_refine else 'on (OpenMM)'}\n")

    try:
        from ImmuneBuilder import NanoBodyBuilder2  # type: ignore
        from ImmuneBuilder.refine import refine      # type: ignore
    except ImportError:
        sys.exit(
            "Error: ImmuneBuilder is not installed.\n"
            "Run:  pip install ImmuneBuilder pdbfixer anarci"
        )

    print("Loading NanobodyBuilder2 models…")
    builder = NanoBodyBuilder2()
    refine_fn = (lambda src, dst: False) if args.no_refine else refine

    total = len(entries)
    failed = []
    for i, (raw_name, sequence) in enumerate(entries, 1):
        name = make_safe_name(raw_name)
        print(f"\n[{i}/{total}] {name}")
        seq = sequence.strip().replace(" ", "").replace("\n", "")
        if not seq:
            print(f"  Skipping — empty sequence.", file=sys.stderr)
            failed.append(name)
            continue

        out_dir = out_root / name
        results = predict_one(name, seq, out_dir, not args.no_refine, builder, refine_fn)
        if not results:
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Done.  {total - len(failed)}/{total} sequences completed successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Structures written to: {out_root}/")


if __name__ == "__main__":
    main()
