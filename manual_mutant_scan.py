#!/usr/bin/env python3
"""
Score manually-specified VHH mutant sequences against a validated WT complex.

Takes a FASTA of one or more manually-designed mutant sequences and a
validated WT complex (--wt-cif, e.g. a best/converged model from
ensemble_pipeline.py), and docks each mutant templated on that same complex
for both chains -- the WT complex fixes the epitope/binding pose, so every
mutant is scored under the same pose/conditions as the WT rather than risking
an independently re-sampled one. This is the same templating strategy
interface_remodeling.py uses for its automated CDR scan, applied here to
sequences you supply directly instead of an Ala/Gly or enhancement panel.

The WT sequence itself (read from chain A of --wt-cif) is also rescored under
identical conditions, giving a binding_score baseline so each mutant's
binding_score_delta is a like-for-like comparison, not an absolute score
against an externally-sourced or unscored reference.

This script does not reimplement docking -- it reuses dock_mutant and the
scoring helpers from interface_remodeling.py, so mutant and WT scores are
computed exactly the same way as the rest of this pipeline.

Usage:
    python manual_mutant_scan.py --fasta my_mutants.fasta --wt-cif validated/Cluster_12_model_1.cif
    python manual_mutant_scan.py --fasta my_mutants.fasta --wt-cif validated/Cluster_12_model_1.cif \\
        --num-models 10 --recycling-steps 5 --output my_mutants_out/

FASTA input:
    One or more mutant VHH sequences, e.g.:
        >I33A_S57Y
        QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMG...
    Each sequence is docked independently, templated on --wt-cif for both
    chains. The FASTA header becomes the variant's name in the report and
    output filenames (spaces and slashes are sanitised).

Key flags:
    --fasta PATH         FASTA file of mutant VHH sequences. Required.
    --wt-cif PATH        Validated WT complex (chain A = VHH, chain B =
                          antigen) used as the fixed pose template for the
                          WT rescore and every mutant. Required.
    --num-models N       Diffusion samples per docking run. Default: 5
    --recycling-steps N  Boltz2 recycling iterations per sample. Default: 5
    --max-parallel-samples N
    --no-msa-server
    --accelerator gpu/cpu/tpu
    --output DIR

Outputs (under --output, default: a directory named after the FASTA's own
basename, in the cwd):
    docking/{variant}/        Boltz2 rigid-body docking results per mutant,
                               plus the WT rescoring run.
    best_structures/           WT.cif (the input --wt-cif, copied in as-is),
                               WT_rescored.cif (its best-scoring model from
                               the WT rescore), and one CIF per mutant named
                               after its FASTA header -- load this directory
                               directly in a structure viewer to compare
                               mutants against the WT.
    scoring_report.txt         Human-readable summary: WT baseline, then one
                               block per mutant with binding_score,
                               binding_score_delta, ipTM, BSA, and clashes,
                               ranked by binding_score_delta.
    logs/                      Boltz2 stdout/stderr logs
"""

import argparse
import sys
from pathlib import Path

from interface_remodeling import (
    best_row,
    copy_structure,
    dock_mutant,
    model_cif_path,
    reference_metrics,
    summarize_rows,
)
from ensemble_pipeline import extract_all_sequences, load_structure, make_safe_name


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dock manually-specified mutant VHH sequences templated on a validated "
            "WT complex, so mutants are scored against the same fixed pose as the "
            "WT rather than an independently re-sampled one."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, metavar="PATH",
                        help="FASTA file of mutant VHH sequences. Required.")
    parser.add_argument("--wt-cif", required=True, metavar="PATH",
                        help=(
                            "Validated WT complex (chain A = VHH, chain B = antigen), "
                            "used as the fixed pose template for the WT rescore and "
                            "every mutant. Required."
                        ))
    parser.add_argument("--num-models", type=int, default=5, metavar="N",
                        help="Diffusion samples per docking run. Default: 5")
    parser.add_argument("--recycling-steps", type=int, default=5, metavar="N",
                        help="Boltz2 recycling iterations per diffusion sample. Default: 5")
    parser.add_argument("--max-parallel-samples", type=int, default=None, metavar="N",
                        help="Maximum diffusion samples to run in parallel on the GPU.")
    parser.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "tpu"],
                        help="Hardware accelerator for Boltz2. Default: gpu")
    parser.add_argument("--no-msa-server", action="store_true",
                        help="Disable the ColabFold MSA server for the antigen chain.")
    parser.add_argument("--output", default=None, metavar="DIR",
                        help=(
                            "Output root directory. Default: a directory named after "
                            "--fasta's basename, in the cwd."
                        ))
    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.exists():
        sys.exit(f"Error: FASTA file not found -- {args.fasta}")

    mutants = parse_fasta(fasta_path)
    if not mutants:
        sys.exit(f"Error: no sequences found in {args.fasta}")

    wt_cif = Path(args.wt_cif)
    if not wt_cif.exists():
        sys.exit(f"Error: WT complex CIF not found -- {args.wt_cif}")

    complex_model = load_structure(wt_cif)
    complex_seqs = extract_all_sequences(complex_model)
    if "A" not in complex_seqs or "B" not in complex_seqs:
        sys.exit(
            f"Error: expected chains 'A' (VHH) and 'B' (antigen) in {args.wt_cif}, "
            f"found: {list(complex_seqs.keys())}"
        )
    wt_seq = complex_seqs["A"]
    antigen_seq = complex_seqs["B"]
    wt_name = wt_cif.stem

    print(f"WT complex: {wt_cif.name}")
    print(f"  Chain A (VHH):     {len(wt_seq)} aa")
    print(f"  Chain B (antigen): {len(antigen_seq)} aa")
    print(f"Loaded {len(mutants)} mutant sequence(s) from {fasta_path.name}\n")

    out_root = Path(args.output) if args.output else Path.cwd() / make_safe_name(fasta_path.stem)
    dock_dir = out_root / "docking"
    log_dir  = out_root / "logs"
    vis_dir  = out_root / "best_structures"
    for d in (dock_dir, log_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    copy_structure(wt_cif, vis_dir, "WT.cif")
    ref_metrics = reference_metrics(wt_cif)

    # WT rescore -- same templated docking conditions as every mutant below, so
    # binding_score_delta compares like with like.
    print("Rescoring WT under mutant docking conditions "
          f"({args.num_models} samples, recycling_steps={args.recycling_steps})...")
    wt_rows = dock_mutant(f"{wt_name}_WT", wt_seq, antigen_seq, wt_cif, dock_dir, log_dir, args)
    if not wt_rows:
        sys.exit("Error: WT rescoring failed (no Boltz2 output).")
    wt_summary = summarize_rows(wt_rows)
    wt_binding_score = wt_summary["mean_binding_score"]
    if wt_binding_score is None:
        sys.exit("Error: WT rescoring produced no binding_score -- check the Boltz2 logs.")

    wt_best = best_row(wt_rows)
    if wt_best is not None:
        wt_safe_name = make_safe_name(f"{wt_name}_WT")
        wt_src = model_cif_path(dock_dir, wt_safe_name, wt_best["Model"])
        copy_structure(wt_src, vis_dir, "WT_rescored.cif")
    print(f"WT rescored: mean_binding_score={wt_binding_score}  "
          f"mean_iptm={wt_summary['mean_iptm']}  mean_bsa_A2={wt_summary['mean_bsa_A2']}\n")

    # Dock each manual mutant, templated on the same WT complex.
    results = []
    for i, (variant_name, mut_seq) in enumerate(mutants, 1):
        print(f"[{i}/{len(mutants)}] {variant_name} ({len(mut_seq)} aa)")
        if len(mut_seq) != len(wt_seq):
            print(f"  WARNING: length differs from WT ({len(mut_seq)} vs {len(wt_seq)} aa) "
                  f"-- Boltz2's template alignment is sequence-based and should still work, "
                  f"but this is not a simple point mutant.")

        rows = dock_mutant(variant_name, mut_seq, antigen_seq, wt_cif, dock_dir, log_dir, args)
        summary = summarize_rows(rows)

        best = best_row(rows)
        if best is not None:
            safe_name = make_safe_name(variant_name)
            src = model_cif_path(dock_dir, safe_name, best["Model"])
            copy_structure(src, vis_dir, f"{make_safe_name(variant_name)}.cif")
            print(f"  Best model (binding_score={best['binding_score']:.3f})")
        else:
            print(f"  No scored model for '{variant_name}'.")

        results.append({
            "Variant": variant_name,
            "sequence": mut_seq,
            "mean_binding_score": summary["mean_binding_score"],
            "binding_score_delta": (
                round(summary["mean_binding_score"] - wt_binding_score, 4)
                if summary["mean_binding_score"] is not None else None
            ),
            "mean_iptm": summary["mean_iptm"],
            "mean_bsa_A2": summary["mean_bsa_A2"],
            "bsa_delta_vs_wt": (
                round(summary["mean_bsa_A2"] - ref_metrics["bsa_A2"], 1)
                if summary["mean_bsa_A2"] is not None and ref_metrics["bsa_A2"] is not None
                else None
            ),
            "total_clashes": summary["total_clashes"],
            "n_docking_models": len(rows),
        })
        print()

    results.sort(key=lambda r: (r["binding_score_delta"] is None, r["binding_score_delta"]))

    # ---- Report ----
    report_path = out_root / "scoring_report.txt"
    lines = []
    lines.append("=" * 70)
    lines.append("MANUAL MUTANT SCORING REPORT")
    lines.append("=" * 70)
    lines.append(f"WT complex:  {wt_cif}")
    lines.append(f"FASTA input: {fasta_path}")
    lines.append(f"Docking:     {args.num_models} diffusion samples, "
                 f"recycling_steps={args.recycling_steps}")
    lines.append("")
    lines.append("WT (rescored, templated on itself):")
    lines.append(f"  mean_binding_score = {wt_binding_score}")
    lines.append(f"  mean_iptm          = {wt_summary['mean_iptm']}")
    lines.append(f"  mean_bsa_A2        = {wt_summary['mean_bsa_A2']}")
    lines.append(f"  total_clashes      = {wt_summary['total_clashes']}")
    lines.append("")
    lines.append(f"WT reference geometry (input --wt-cif): bsa_A2={ref_metrics['bsa_A2']}  "
                 f"n_clashes={ref_metrics['n_clashes']}")
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"{len(results)} mutant(s), ranked by binding_score_delta "
                 "(ascending -- biggest drops first):")
    lines.append("-" * 70)
    for r in results:
        lines.append("")
        lines.append(f"{r['Variant']}")
        lines.append(f"  mean_binding_score  = {r['mean_binding_score']}")
        lines.append(f"  binding_score_delta = {r['binding_score_delta']}  (vs WT rescore)")
        lines.append(f"  mean_iptm           = {r['mean_iptm']}")
        lines.append(f"  mean_bsa_A2         = {r['mean_bsa_A2']}  "
                     f"(delta vs WT: {r['bsa_delta_vs_wt']})")
        lines.append(f"  total_clashes       = {r['total_clashes']}")
        lines.append(f"  n_docking_models    = {r['n_docking_models']}")
    lines.append("")
    lines.append("=" * 70)
    report_path.write_text("\n".join(lines) + "\n")

    print(f"{'='*70}")
    print(f"Done. {len(results)} mutant(s) scored.")
    print(f"Report: {report_path}")
    print(f"Structures for visualisation: {vis_dir}")


if __name__ == "__main__":
    main()
