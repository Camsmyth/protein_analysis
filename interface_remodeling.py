#!/usr/bin/env python3
"""
CDR interface alanine/glycine-scan remodeling for a single VHH.

Takes one VHH (already triaged by ensemble_pipeline.py) and proposes point
mutations at its antigen-contacting CDR residues, then re-predicts each
mutant through the same NanobodyBuilder2 -> Boltz2 rigid-body docking
pipeline to score whether the mutation improves or degrades binding.

Pipeline:
  Stage 0  ANARCI (IMGT scheme) numbers the VHH and locates CDR1/2/3 vs
           framework boundaries.
  Stage 1  Baseline: NanobodyBuilder2 rank-0 model + Boltz2 docking of the
           wild-type sequence against the antigen (same as ensemble_pipeline.py
           Stages 1-2). Antigen-contacting residues are read off this baseline.
  Stage 2  Mutation generation: every wild-type CDR residue found at the
           interface is scanned to alanine (or glycine, if the wild-type
           residue is already alanine).
  Stage 3  Each mutant sequence is re-run through NanobodyBuilder2 + Boltz2
           docking (identical settings to the baseline) and scored.
  Stage 4  Deltas vs. the wild-type baseline are reported, ranked by
           binding_score_delta.

This script does not reimplement docking or structure prediction -- it reuses
run_immunebuilder_best, dock_vhh, and the metric helpers from
ensemble_pipeline.py so mutant and wild-type scores are directly comparable.

Usage:
    python interface_remodeling.py --sequence QVQLVESGG... --antigen h7/hCD7_alphafold.pdb
    python interface_remodeling.py --name Cluster_12 \\
        --input enriched_clusters.csv --antigen h7/hCD7_alphafold.pdb \\
        --use-template --num-models 5

Key flags:
    --sequence SEQ          VHH amino acid sequence (mutually exclusive with --name/--input)
    --name NAME             Cluster name to look up in --input (requires --input)
    --input PATH            CSV/XLSX containing --name (same schema as ensemble_pipeline.py)
    --antigen PATH          Antigen structure file (.pdb or .cif) [required]
    --antigen-chain ID      Single chain from antigen (default: first)
    --antigen-chains ID ... Multiple chains merged into one
    --use-template          Provide antigen as a Boltz2 structural template
    --cdrs 1 2 3            Which CDR loops to scan (default: all three)
    --num-models N          Diffusion samples per docking run (default: 5)
    --recycling-steps N     Boltz2 recycling iterations per sample (default: 3)
    --max-parallel-samples N
    --no-msa-server
    --output DIR
    --accelerator gpu/cpu/tpu

Outputs (under --output):
    vhh_structures/{variant}/     NanobodyBuilder2 rank-0 PDB + CIF (WT + each mutant)
    docking/{variant}/            Boltz2 rigid-body docking results
    mutation_candidates.csv       One row per mutant, ranked by binding_score_delta
    logs/                         Boltz2 stdout/stderr logs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from ensemble_pipeline import (
    binding_score,
    clean_sequence,
    dock_vhh,
    extract_all_sequences,
    load_structure,
    make_safe_name,
    merge_antigen_chains,
    prepare_antigen_cif,
    run_immunebuilder_best,
)

# IMGT scheme CDR boundaries (inclusive), per the standard VHH/VH numbering.
IMGT_CDR_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}


# ---------------------------------------------------------------------------
# Stage 0 -- ANARCI numbering / CDR localisation
# ---------------------------------------------------------------------------

def number_vhh(sequence: str) -> list[tuple[int, str]]:
    """
    IMGT-number a VHH sequence with ANARCI.

    Returns a list of (imgt_position, residue) for each non-gap residue, in
    the same left-to-right order as `sequence`. Raises RuntimeError if ANARCI
    finds no antibody variable domain in the sequence.
    """
    try:
        import anarci
    except ImportError:
        sys.exit("Error: anarci is not installed.\nRun:  pip install anarci")

    _, numbered, _, _ = anarci.run_anarci([("query", sequence)], scheme="imgt", output=False)
    domains = numbered[0]
    if not domains:
        raise RuntimeError("ANARCI found no antibody variable domain in this sequence.")

    numbered_seq, _, _ = domains[0]
    return [(pos, aa) for (pos, _ins), aa in numbered_seq if aa != "-"]


def assign_cdr_regions(numbering: list[tuple[int, str]]) -> dict[int, str]:
    """Map each sequence index (0-based, into the original sequence) to a region label."""
    regions: dict[int, str] = {}
    for seq_idx, (imgt_pos, _aa) in enumerate(numbering):
        region = "FR"
        for cdr_name, (lo, hi) in IMGT_CDR_RANGES.items():
            if lo <= imgt_pos <= hi:
                region = cdr_name
                break
        regions[seq_idx] = region
    return regions


# ---------------------------------------------------------------------------
# Interface residue parsing (matches ensemble_pipeline.py's contact format)
# ---------------------------------------------------------------------------

def parse_binder_interface_residues(s: object) -> set[int]:
    """Parse 'N (Q3,S27,...)' -> {3, 27, ...} 1-based binder residue numbers."""
    import re
    m = re.match(r"\d+\s*\((.+)\)", str(s or ""))
    if not m:
        return set()
    result = set()
    for token in m.group(1).split(","):
        token = token.strip()
        if token and len(token) >= 2:
            try:
                result.add(int(token[1:]))
            except ValueError:
                pass
    return result


# ---------------------------------------------------------------------------
# Stage 2 -- mutation generation
# ---------------------------------------------------------------------------

def generate_ala_scan_mutants(
    sequence: str,
    regions: dict[int, str],
    interface_positions_1based: set[int],
    cdrs_to_scan: set[str],
) -> list[tuple[str, int, str, str, str]]:
    """
    Build one alanine (or glycine) point mutant per wild-type CDR residue at
    the interface.

    Returns a list of (variant_name, seq_idx_0based, wt_aa, mut_aa, cdr_label).
    """
    mutants = []
    for seq_idx, wt_aa in enumerate(sequence):
        pos_1based = seq_idx + 1
        if pos_1based not in interface_positions_1based:
            continue
        region = regions.get(seq_idx, "FR")
        if region not in cdrs_to_scan:
            continue
        mut_aa = "G" if wt_aa == "A" else "A"
        variant_name = f"{wt_aa}{pos_1based}{mut_aa}"
        mutants.append((variant_name, seq_idx, wt_aa, mut_aa, region))
    return mutants


def apply_mutation(sequence: str, seq_idx: int, mut_aa: str) -> str:
    return sequence[:seq_idx] + mut_aa + sequence[seq_idx + 1:]


# ---------------------------------------------------------------------------
# Docking wrapper (baseline + each mutant go through the same code path)
# ---------------------------------------------------------------------------

def predict_and_dock(
    variant_label: str,
    sequence: str,
    antigen_seq: str,
    antigen_cif: Path,
    struct_dir: Path,
    dock_dir: Path,
    log_dir: Path,
    args,
) -> list[dict]:
    """Run NanobodyBuilder2 + Boltz2 docking for one sequence variant. Returns per-model rows."""
    safe_name = make_safe_name(variant_label)
    name_struct_dir = struct_dir / safe_name
    name_struct_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n-- {variant_label}: Stage 1 (NanobodyBuilder2) --")
    best_model_cif = run_immunebuilder_best(sequence, safe_name, name_struct_dir)
    if best_model_cif is None:
        print(f"  No VHH structure generated for '{variant_label}' -- skipping.")
        return []

    print(f"-- {variant_label}: Stage 2 (Boltz2 docking, "
          f"{args.num_models} samples, recycling_steps={args.recycling_steps}) --")
    rows = dock_vhh(
        name=safe_name,
        best_model_cif=best_model_cif,
        binder_seq=sequence,
        antigen_seq=antigen_seq,
        antigen_cif=antigen_cif,
        dock_root=dock_dir,
        accelerator=args.accelerator,
        num_models=args.num_models,
        no_msa_server=args.no_msa_server,
        use_template=args.use_template,
        max_parallel_samples=args.max_parallel_samples,
        log_dir=log_dir,
        recycling_steps=args.recycling_steps,
    )
    for row in rows:
        row["variant"] = variant_label
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    """Mean binding_score / iptm / bsa_A2 / n_clashes across diffusion samples."""
    if not rows:
        return {
            "mean_binding_score": None, "mean_iptm": None,
            "mean_bsa_A2": None, "total_clashes": None,
        }
    scores  = [r["binding_score"] for r in rows if r.get("binding_score") is not None]
    iptms   = [r["iptm"] for r in rows if r.get("iptm") is not None]
    bsas    = [r["bsa_A2"] for r in rows if r.get("bsa_A2") is not None]
    clashes = sum(r.get("n_clashes") or 0 for r in rows)
    return {
        "mean_binding_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_iptm":          round(sum(iptms) / len(iptms), 4) if iptms else None,
        "mean_bsa_A2":        round(sum(bsas) / len(bsas), 1) if bsas else None,
        "total_clashes":      clashes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Alanine/glycine-scan remodeling of a VHH's antigen-contacting CDR "
            "residues, re-predicted through NanobodyBuilder2 + Boltz2 rigid-body "
            "docking and scored against the wild-type baseline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    seqsrc = parser.add_argument_group("VHH sequence (choose one)")
    seqsrc.add_argument("--sequence", default=None, metavar="SEQ",
                        help="VHH amino acid sequence.")
    seqsrc.add_argument("--name", default=None, metavar="NAME",
                        help="Cluster name to look up in --input.")
    seqsrc.add_argument("--input", default=None, metavar="PATH",
                        help="CSV/XLSX containing --name (same schema as ensemble_pipeline.py).")
    seqsrc.add_argument("--names", default="Cluster", metavar="COLUMN",
                        help="Sample name column in --input. Default: Cluster")
    seqsrc.add_argument("--sequences", default="Protein_Sequence_R2", metavar="COLUMN",
                        help="Sequence column in --input. Default: Protein_Sequence_R2")

    ag = parser.add_argument_group("Antigen")
    ag.add_argument("--antigen", required=True, metavar="PATH",
                    help="Antigen structure file (.pdb or .cif).")
    ag.add_argument("--antigen-chain", default=None, metavar="ID",
                    help="Single chain to use from the antigen structure (default: first).")
    ag.add_argument("--antigen-chains", nargs="+", default=None, metavar="ID",
                    help="Multiple antigen chains, merged into one for docking.")
    ag.add_argument("--use-template", action="store_true",
                    help="Provide the antigen as a Boltz2 structural template for chain B.")

    scan = parser.add_argument_group("Mutation scan")
    scan.add_argument("--cdrs", nargs="+", default=["1", "2", "3"], choices=["1", "2", "3"],
                      metavar="N",
                      help="Which CDR loop(s) to scan for interface mutations. Default: 1 2 3")

    dock = parser.add_argument_group("Docking (same conventions as ensemble_pipeline.py)")
    dock.add_argument("--num-models", type=int, default=5, metavar="N",
                      help="Diffusion samples per docking run (baseline and each mutant). Default: 5")
    dock.add_argument("--recycling-steps", type=int, default=3, metavar="N",
                      help="Boltz2 recycling iterations per diffusion sample. Default: 3")
    dock.add_argument("--max-parallel-samples", type=int, default=None, metavar="N",
                      help="Maximum diffusion samples to run in parallel on the GPU.")

    hw = parser.add_argument_group("Hardware and MSA")
    hw.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "tpu"],
                    help="Hardware accelerator for Boltz2. Default: gpu")
    hw.add_argument("--no-msa-server", action="store_true",
                    help="Disable the ColabFold MSA server.")

    out = parser.add_argument_group("Output")
    out.add_argument("--output", default=None, metavar="DIR",
                     help="Output root directory. Default: interface_remodeling/ in the cwd.")

    args = parser.parse_args()

    # ---- Resolve the wild-type VHH sequence ----
    if args.sequence:
        wt_name = "query"
        wt_seq = clean_sequence(args.sequence)
    elif args.name and args.input:
        df = pd.read_csv(args.input) if str(args.input).lower().endswith(".csv") \
            else pd.read_excel(args.input)
        matches = df[df[args.names].astype(str) == str(args.name)]
        if matches.empty:
            sys.exit(f"Error: name '{args.name}' not found in column '{args.names}' of {args.input}")
        wt_name = str(args.name)
        wt_seq = clean_sequence(matches.iloc[0][args.sequences])
    else:
        sys.exit("Error: provide either --sequence, or both --name and --input.")

    print(f"Wild-type VHH: {wt_name} ({len(wt_seq)} aa)")
    print(f"  {wt_seq}\n")

    # ---- ANARCI numbering ----
    print("Stage 0: ANARCI (IMGT) numbering...")
    numbering = number_vhh(wt_seq)
    regions = assign_cdr_regions(numbering)
    cdrs_to_scan = {f"CDR{n}" for n in args.cdrs}
    cdr_counts = {c: sum(1 for r in regions.values() if r == c) for c in ["CDR1", "CDR2", "CDR3"]}
    print(f"  CDR lengths: CDR1={cdr_counts['CDR1']}  CDR2={cdr_counts['CDR2']}  "
          f"CDR3={cdr_counts['CDR3']}  (scanning: {', '.join(sorted(cdrs_to_scan))})\n")

    # ---- Antigen setup (mirrors ensemble_pipeline.py) ----
    antigen_path = Path(args.antigen)
    if not antigen_path.exists():
        sys.exit(f"Error: antigen file not found -- {args.antigen}")

    antigen_model = load_structure(antigen_path)
    antigen_seqs_all = extract_all_sequences(antigen_model)
    if not antigen_seqs_all:
        sys.exit("Error: no protein chains found in antigen structure.")

    requested_chains = args.antigen_chains or ([args.antigen_chain] if args.antigen_chain else None)
    if requested_chains:
        missing = [c for c in requested_chains if c not in antigen_seqs_all]
        if missing:
            sys.exit(f"Error: chain(s) {missing} not found. Available: {list(antigen_seqs_all.keys())}")
        if len(requested_chains) == 1:
            antigen_seq = antigen_seqs_all[requested_chains[0]]
            antigen_cif = prepare_antigen_cif(antigen_path)
        else:
            antigen_cif, antigen_seq = merge_antigen_chains(antigen_path, requested_chains)
    else:
        _, antigen_seq = next(iter(antigen_seqs_all.items()))
        antigen_cif = prepare_antigen_cif(antigen_path)

    print(f"Antigen: {antigen_path.name}  ({len(antigen_seq)} residues)\n")

    # ---- Output directories ----
    out_root = Path(args.output) if args.output else Path.cwd() / "interface_remodeling"
    struct_dir = out_root / "vhh_structures"
    dock_dir   = out_root / "docking"
    log_dir    = out_root / "logs"
    for d in (struct_dir, dock_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: wild-type baseline ----
    wt_label = f"{wt_name}_WT"
    wt_rows = predict_and_dock(
        wt_label, wt_seq, antigen_seq, antigen_cif, struct_dir, dock_dir, log_dir, args,
    )
    if not wt_rows:
        sys.exit(f"Error: wild-type baseline prediction/docking failed for '{wt_name}'.")

    wt_summary = summarize_rows(wt_rows)
    print(f"\nWild-type baseline: mean_binding_score={wt_summary['mean_binding_score']}  "
          f"mean_iptm={wt_summary['mean_iptm']}  mean_bsa_A2={wt_summary['mean_bsa_A2']}")

    interface_positions: set[int] = set()
    for row in wt_rows:
        interface_positions |= parse_binder_interface_residues(row.get("binder_interface_residues"))
    if not interface_positions:
        sys.exit(
            "Error: no binder interface residues detected in the wild-type baseline docking "
            "(check n_clashes / binding_score -- the baseline pose may not be usable)."
        )
    print(f"Wild-type interface residues (binder, 1-based): {sorted(interface_positions)}\n")

    # ---- Stage 2: generate mutants ----
    mutants = generate_ala_scan_mutants(wt_seq, regions, interface_positions, cdrs_to_scan)
    if not mutants:
        sys.exit(
            "No CDR interface residues found to scan -- either the interface doesn't touch "
            "the selected CDR(s), or --cdrs excludes the contacting loop(s)."
        )
    print(f"Stage 2: {len(mutants)} candidate mutation(s) to scan: "
          f"{', '.join(m[0] for m in mutants)}\n")

    # ---- Stage 3: re-predict + dock each mutant ----
    results = []
    for i, (variant_name, seq_idx, wt_aa, mut_aa, cdr_label) in enumerate(mutants, 1):
        mut_seq = apply_mutation(wt_seq, seq_idx, mut_aa)
        variant_label = f"{wt_name}_{variant_name}"
        print(f"\n[{i}/{len(mutants)}] {variant_label}  ({cdr_label}, position {seq_idx + 1}: "
              f"{wt_aa}->{mut_aa})")
        rows = predict_and_dock(
            variant_label, mut_seq, antigen_seq, antigen_cif, struct_dir, dock_dir, log_dir, args,
        )
        summary = summarize_rows(rows)

        def _delta(key):
            if summary[key] is None or wt_summary[key] is None:
                return None
            return round(summary[key] - wt_summary[key], 4)

        results.append({
            "Parent": wt_name,
            "Variant": variant_name,
            "CDR": cdr_label,
            "Position": seq_idx + 1,
            "WT_residue": wt_aa,
            "Mut_residue": mut_aa,
            "mean_binding_score": summary["mean_binding_score"],
            "binding_score_delta": _delta("mean_binding_score"),
            "mean_iptm": summary["mean_iptm"],
            "iptm_delta": _delta("mean_iptm"),
            "mean_bsa_A2": summary["mean_bsa_A2"],
            "bsa_delta": _delta("mean_bsa_A2"),
            "total_clashes": summary["total_clashes"],
            "n_docking_models": len(rows),
        })

    # ---- Stage 4: report ----
    out_df = pd.DataFrame(results)
    if not out_df.empty:
        out_df = out_df.sort_values("binding_score_delta", ascending=False, na_position="last")
    out_csv = out_root / "mutation_candidates.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"Done. {len(results)}/{len(mutants)} mutants scored.")
    print(f"Wild-type mean_binding_score: {wt_summary['mean_binding_score']}")
    if not out_df.empty:
        print("\nTop candidates by binding_score_delta:")
        cols = ["Variant", "CDR", "mean_binding_score", "binding_score_delta", "total_clashes"]
        print(out_df[cols].head(10).to_string(index=False))
    print(f"\nFull results: {out_csv}")


if __name__ == "__main__":
    main()
