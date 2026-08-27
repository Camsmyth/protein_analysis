#!/usr/bin/env python3
"""
CDR interface alanine/glycine-scan remodeling from a validated docked complex.

Takes an already-validated, high-confidence VHH-antigen complex (e.g. a
structure that passed ensemble_pipeline.py's READY tier / convergence
scoring) and proposes point mutations at its antigen-contacting CDR
residues. Each mutant is re-predicted templated on that same validated
complex for BOTH chains, so mutants are scored on the same fixed epitope
and binding pose as the input rather than risking a fresh, possibly wrong,
pose from scratch.

This intentionally does not re-dock a wild-type baseline with Boltz2: Boltz2
diffusion is not guaranteed to converge on the correct pose, so re-deriving
the baseline here would let a bad WT pose propagate into every mutant's
score. Instead, the input --complex-cif is trusted as ground truth for the
epitope and binding mode, and every mutant is anchored to it.

Pipeline:
  Stage 0  Load --complex-cif, extract chain A (binder) and chain B
           (antigen) sequences, and ANARCI (IMGT scheme) number the binder
           to locate CDR1/2/3 vs framework.
  Stage 1  Interface residues are read directly off --complex-cif (no
           docking) via ensemble_pipeline.py's contact/BSA/clash geometry
           functions. Used later for round-2 classification, not as a
           pre-filter on round 1.
  Stage 2  Mutation generation: every residue in the selected CDR loop(s) is
           scanned to alanine (or glycine, if already alanine) -- the whole
           CDR panel, not just residues already flagged as interface-
           contacting, so a position whose small WT sidechain doesn't
           currently reach the antigen is still tested.
  Stage 3  Each mutant sequence is docked with Boltz2, templated on
           --complex-cif for BOTH chain A and chain B (force=False, so the
           template is a strong structural prior, not a hard constraint --
           diffusion still resolves the local change at the mutated
           position). No fresh NanobodyBuilder2 run per mutant is needed
           since the mutant differs from the template by one residue and
           Boltz2's template alignment is sequence-based, not exact-match.
  Stage 4  Results are reported, ranked by mean_binding_score.
  Stage 5  (--optimize only) Each round-1 position is classified as
           hotspot (interface-contacting, score drops well below the round-1
           panel median -- left alone), tolerant_contact (interface-
           contacting, score barely moved), or cold_spot (not currently
           interface-contacting at all). A second round tries a panel of
           paratope-enriched substitutions (Tyr/Arg/Trp/Asp/Asn) at every
           tolerant_contact and cold_spot position, looking for a bulkier or
           more interactive sidechain that creates a new favourable contact
           -- skipping confirmed hotspots, which the scan already showed are
           contributing and best left alone.

This script does not reimplement docking or structure prediction -- it
reuses dock_vhh and the metric helpers from ensemble_pipeline.py so mutant
and reference scores are directly comparable.

Usage:
    python interface_remodeling.py --complex-cif validated/Cluster_12_model_1.cif
    python interface_remodeling.py --complex-cif validated/Cluster_12_model_1.cif \\
        --wt-binding-score 0.812 --optimize

Key flags:
    --complex-cif PATH      Validated two-chain VHH+antigen complex (chain A =
                             binder, chain B = antigen). Required.
    --wt-binding-score SCORE  Known WT binding_score (e.g. from
                               ensemble_binding_scores.csv) to compute
                               binding_score_delta per mutant. Optional --
                               without it, only absolute mean_binding_score
                               is reported (a static reference CIF has no
                               Boltz2 confidence JSON of its own).
    --cdrs 1 2 3             Which CDR loops to scan (default: all three)
    --optimize               Run the round-2 enhancement panel (see Stage 5)
    --hotspot-drop-fraction F  Hotspot classification threshold (default: 0.15)
    --num-models N           Diffusion samples per mutant docking run (default: 5)
    --recycling-steps N      Boltz2 recycling iterations per sample (default: 3)
    --max-parallel-samples N
    --no-msa-server
    --output DIR
    --accelerator gpu/cpu/tpu

Outputs (under --output, which defaults to a directory named after the CIF's
own basename, e.g. ./Cluster_12_model_1/):
    docking/{variant}/       Boltz2 rigid-body docking results per mutant
    best_structures/          WT.cif (the input --complex-cif, copied in as the
                               reference) plus one CIF per mutant named after
                               its single-residue mutation, e.g. I33A.cif --
                               load this directory directly in a structure
                               viewer to compare mutants against the WT.
                               Override with --vis-dir.
    mutation_candidates.csv   One row per mutant (round1, and round2 if
                               --optimize was used). Ranked by
                               binding_score_delta if --wt-binding-score was
                               given (ascending -- biggest drops first),
                               otherwise by mean_binding_score (descending).
                               Round column and Position_label
                               (hotspot/tolerant_contact/cold_spot, round2
                               rows only) identify which stage/classification
                               each row came from.
    logs/                     Boltz2 stdout/stderr logs
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from ensemble_pipeline import (
    binding_score,
    calculate_bsa,
    count_heavy_atom_clashes,
    count_interface_contacts,
    dock_vhh,
    extract_all_sequences,
    load_structure,
    make_safe_name,
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
# Stage 1 -- interface residues, read directly off the validated complex
# ---------------------------------------------------------------------------

def interface_positions_from_complex(complex_cif: Path) -> set[int]:
    """1-based binder (chain A) residue numbers in contact with the antigen (chain B)."""
    import re
    contacts = count_interface_contacts(complex_cif)
    s = contacts.get("binder_interface_residues")
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
    cdrs_to_scan: set[str],
) -> list[tuple[str, int, str, str, str]]:
    """
    Build one alanine (or glycine) point mutant per wild-type residue in the
    selected CDR loop(s) -- the full CDR panel, not just residues already
    flagged as antigen-contacting in the reference complex. Restricting to
    interface-only residues would bias round 1 against ever discovering a CDR
    position whose current (small) sidechain doesn't reach the antigen but
    could if swapped for something bulkier -- exactly the "cold spot" case a
    later round is meant to probe.

    Returns a list of (variant_name, seq_idx_0based, wt_aa, mut_aa, cdr_label).
    """
    mutants = []
    for seq_idx, wt_aa in enumerate(sequence):
        region = regions.get(seq_idx, "FR")
        if region not in cdrs_to_scan:
            continue
        pos_1based = seq_idx + 1
        mut_aa = "G" if wt_aa == "A" else "A"
        variant_name = f"{wt_aa}{pos_1based}{mut_aa}"
        mutants.append((variant_name, seq_idx, wt_aa, mut_aa, region))
    return mutants


def apply_mutation(sequence: str, seq_idx: int, mut_aa: str) -> str:
    return sequence[:seq_idx] + mut_aa + sequence[seq_idx + 1:]


# ---------------------------------------------------------------------------
# Stage 5 -- hotspot classification + second-round enhancing substitutions
# ---------------------------------------------------------------------------

# Paratope-enriched residues to try at positions the Ala/Gly scan showed are
# not currently pulling their weight -- aromatics/H-bonders/charged residues
# over-represented in antibody CDR loops (Tyr, Trp, Arg, Asp, Asn).
ENHANCEMENT_PANEL = ["Y", "R", "W", "D", "S", "H", "S", "Q","T","G"]

# A position's Ala/Gly mutant scoring this far below the round-1 panel median
# marks it a hotspot (mutating further is not attempted). Relative to the
# panel's own median rather than an absolute cutoff, since binding_score scale
# varies VHH to VHH.
_HOTSPOT_DROP_FRACTION = 0.10


def classify_positions(
    round1_results: list[dict],
    interface_positions_1based: set[int],
    hotspot_drop_fraction: float = _HOTSPOT_DROP_FRACTION,
) -> dict[int, str]:
    """
    Label each scanned position as one of:
      "hotspot"           -- interface-contacting, Ala/Gly score drops well
                              below the round-1 panel median. Leave alone.
      "tolerant_contact"  -- interface-contacting, but the Ala/Gly swap barely
                              hurt binding_score. Candidate for round 2.
      "cold_spot"         -- not currently antigen-contacting at all in the
                              reference complex. Candidate for round 2.

    Positions with no round-1 score (docking failed) are omitted.
    """
    scored = [r for r in round1_results if r.get("mean_binding_score") is not None]
    if not scored:
        return {}
    scores = sorted(r["mean_binding_score"] for r in scored)
    n = len(scores)
    median = scores[n // 2] if n % 2 else (scores[n // 2 - 1] + scores[n // 2]) / 2
    hotspot_bar = median * (1 - hotspot_drop_fraction)

    labels: dict[int, str] = {}
    for r in scored:
        pos = r["Position"]
        if pos not in interface_positions_1based:
            labels[pos] = "cold_spot"
        elif r["mean_binding_score"] < hotspot_bar:
            labels[pos] = "hotspot"
        else:
            labels[pos] = "tolerant_contact"
    return labels


def generate_enhancement_mutants(
    sequence: str,
    regions: dict[int, str],
    labels: dict[int, str],
) -> list[tuple[str, int, str, str, str, str]]:
    """
    Build one point mutant per (non-hotspot position, enhancement residue)
    pair, skipping the WT residue itself.

    Returns a list of (variant_name, seq_idx_0based, wt_aa, mut_aa, cdr_label, position_label).
    """
    mutants = []
    for seq_idx, wt_aa in enumerate(sequence):
        pos_1based = seq_idx + 1
        label = labels.get(pos_1based)
        if label not in ("tolerant_contact", "cold_spot"):
            continue
        region = regions.get(seq_idx, "FR")
        for mut_aa in ENHANCEMENT_PANEL:
            if mut_aa == wt_aa:
                continue
            variant_name = f"{wt_aa}{pos_1based}{mut_aa}"
            mutants.append((variant_name, seq_idx, wt_aa, mut_aa, region, label))
    return mutants


# ---------------------------------------------------------------------------
# Stage 3 -- mutant docking, templated on the validated complex for both chains
# ---------------------------------------------------------------------------

def dock_mutant(
    variant_label: str,
    mut_seq: str,
    antigen_seq: str,
    complex_cif: Path,
    dock_dir: Path,
    log_dir: Path,
    args,
) -> list[dict]:
    """
    Dock one mutant sequence, templated on complex_cif for both chain A and
    chain B, so the mutant's pose stays anchored to the validated reference
    epitope/orientation rather than being independently re-sampled.
    """
    safe_name = make_safe_name(variant_label)
    print(f"\n-- {variant_label}: Boltz2 docking (templated on reference complex, "
          f"{args.num_models} samples, recycling_steps={args.recycling_steps}) --")
    rows = dock_vhh(
        name=safe_name,
        best_model_cif=complex_cif,
        binder_seq=mut_seq,
        antigen_seq=antigen_seq,
        antigen_cif=complex_cif,
        dock_root=dock_dir,
        accelerator=args.accelerator,
        num_models=args.num_models,
        no_msa_server=args.no_msa_server,
        use_template=True,
        max_parallel_samples=args.max_parallel_samples,
        log_dir=log_dir,
        recycling_steps=args.recycling_steps,
    )
    for row in rows:
        row["variant"] = variant_label
    return rows


def model_cif_path(dock_dir: Path, safe_name: str, model_num: int) -> Path:
    return dock_dir / safe_name / f"{safe_name}_model_{model_num}.cif"


def best_row(rows: list[dict]) -> dict | None:
    """Highest binding_score row, same convention as ensemble_pipeline.py's _find_best_cif."""
    scored = [r for r in rows if r.get("binding_score") is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: r["binding_score"])


def copy_structure(src: Path, dest_dir: Path, dest_name: str) -> Path | None:
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dest_name
    shutil.copy2(src, dest)
    return dest


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


def reference_metrics(complex_cif: Path) -> dict:
    """binding_score/iptm are not available for a raw complex CIF (no confidence JSON) --
    only geometric metrics (BSA, clashes) can be recomputed directly from the structure."""
    return {
        "bsa_A2": calculate_bsa(complex_cif),
        "n_clashes": count_heavy_atom_clashes(complex_cif),
    }


def run_mutant_panel(
    mutants: list[tuple],
    wt_seq: str,
    wt_name: str,
    antigen_seq: str,
    complex_cif: Path,
    dock_dir: Path,
    log_dir: Path,
    vis_dir: Path,
    ref_metrics: dict,
    args,
    round_label: str,
    wt_binding_score: float | None = None,
) -> list[dict]:
    """
    Dock, score, and copy the best structure for every mutant in `mutants`.
    Shared by round 1 (Ala/Gly scan) and round 2 (enhancement panel) -- each
    mutant tuple's first five fields are always
    (variant_name, seq_idx, wt_aa, mut_aa, cdr_label); a sixth optional field
    (position_label) is carried through to the results if present.

    wt_binding_score (from --wt-binding-score) is independent of hotspot
    classification (see classify_positions, which derives its own bar from
    the round-1 panel's median) -- it only drives the binding_score_delta
    column here, so --optimize still works without a known WT score.
    """
    results = []
    for i, mutant in enumerate(mutants, 1):
        variant_name, seq_idx, wt_aa, mut_aa, cdr_label = mutant[:5]
        position_label = mutant[5] if len(mutant) > 5 else None

        mut_seq = apply_mutation(wt_seq, seq_idx, mut_aa)
        variant_label = f"{wt_name}_{variant_name}"
        print(f"\n[{round_label} {i}/{len(mutants)}] {variant_label}  "
              f"({cdr_label}, position {seq_idx + 1}: {wt_aa}->{mut_aa})")
        rows = dock_mutant(
            variant_label, mut_seq, antigen_seq, complex_cif, dock_dir, log_dir, args,
        )
        summary = summarize_rows(rows)

        best = best_row(rows)
        if best is not None:
            mut_safe_name = make_safe_name(variant_label)
            src = model_cif_path(dock_dir, mut_safe_name, best["Model"])
            dest_name = f"{make_safe_name(variant_name)}.cif"
            copied = copy_structure(src, vis_dir, dest_name)
            print(f"  Best model (binding_score={best['binding_score']:.3f}) -> "
                  f"{copied if copied else '[copy failed: ' + str(src) + ' missing]'}")
        else:
            print(f"  No scored model for '{variant_label}' -- nothing copied to {vis_dir}.")

        results.append({
            "Round": round_label,
            "Reference": wt_name,
            "Variant": variant_name,
            "CDR": cdr_label,
            "Position": seq_idx + 1,
            "Position_label": position_label,
            "WT_residue": wt_aa,
            "Mut_residue": mut_aa,
            "mean_binding_score": summary["mean_binding_score"],
            "binding_score_delta": (
                round(summary["mean_binding_score"] - wt_binding_score, 4)
                if summary["mean_binding_score"] is not None and wt_binding_score is not None
                else None
            ),
            "mean_iptm": summary["mean_iptm"],
            "mean_bsa_A2": summary["mean_bsa_A2"],
            "bsa_delta_vs_reference": (
                round(summary["mean_bsa_A2"] - ref_metrics["bsa_A2"], 1)
                if summary["mean_bsa_A2"] is not None and ref_metrics["bsa_A2"] is not None
                else None
            ),
            "total_clashes": summary["total_clashes"],
            "n_docking_models": len(rows),
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Alanine/glycine-scan remodeling of a VHH's antigen-contacting CDR "
            "residues, starting from an already-validated docked complex rather "
            "than a fresh Boltz2 baseline. Each mutant is re-predicted templated "
            "on that same complex for both chains, anchoring it to the validated "
            "epitope/pose instead of risking an independently re-sampled one."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ref = parser.add_argument_group("Reference structure")
    ref.add_argument("--complex-cif", required=True, metavar="PATH",
                     help=(
                         "Validated two-chain VHH-antigen complex (chain A = binder, "
                         "chain B = antigen), e.g. a best/converged model from "
                         "ensemble_pipeline.py's docking output. Used as the fixed "
                         "reference pose for interface detection and as a template "
                         "for every mutant."
                     ))
    ref.add_argument("--wt-binding-score", type=float, default=None, metavar="SCORE",
                     help=(
                         "Known binding_score for the wild-type complex (e.g. from the "
                         "ensemble_binding_scores.csv row this --complex-cif came from). "
                         "If given, each mutant's mutation_candidates.csv row includes "
                         "binding_score_delta = mean_binding_score - this value. If "
                         "omitted, binding_score_delta is left blank -- a static reference "
                         "CIF has no Boltz2 confidence JSON, so there is no WT binding_score "
                         "to diff against unless one is supplied here."
                     ))

    scan = parser.add_argument_group("Mutation scan")
    scan.add_argument("--cdrs", nargs="+", default=["1", "2", "3"], choices=["1", "2", "3"],
                      metavar="N",
                      help="Which CDR loop(s) to scan. Default: 1 2 3")
    scan.add_argument("--optimize", action="store_true",
                      help=(
                          "After the round-1 Ala/Gly scan, classify each scanned position as "
                          "hotspot / tolerant_contact / cold_spot (see classify_positions), "
                          "then run a round-2 panel of paratope-enriched substitutions "
                          f"({', '.join(ENHANCEMENT_PANEL)}) at every tolerant_contact and "
                          "cold_spot position, skipping confirmed hotspots. Multiplies "
                          "compute by roughly the number of qualifying positions x "
                          f"{len(ENHANCEMENT_PANEL)}."
                      ))
    scan.add_argument("--hotspot-drop-fraction", type=float, default=_HOTSPOT_DROP_FRACTION,
                      metavar="F",
                      help=(
                          "A round-1 mutant scoring more than this fraction below the "
                          "round-1 panel median marks its position a hotspot (excluded from "
                          f"--optimize). Default: {_HOTSPOT_DROP_FRACTION}"
                      ))

    dock = parser.add_argument_group("Docking (same conventions as ensemble_pipeline.py)")
    dock.add_argument("--num-models", type=int, default=5, metavar="N",
                      help="Diffusion samples per mutant docking run. Default: 5")
    dock.add_argument("--recycling-steps", type=int, default=3, metavar="N",
                      help="Boltz2 recycling iterations per diffusion sample. Default: 3")
    dock.add_argument("--max-parallel-samples", type=int, default=None, metavar="N",
                      help="Maximum diffusion samples to run in parallel on the GPU.")

    hw = parser.add_argument_group("Hardware and MSA")
    hw.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "tpu"],
                    help="Hardware accelerator for Boltz2. Default: gpu")
    hw.add_argument("--no-msa-server", action="store_true",
                    help="Disable the ColabFold MSA server for the antigen chain.")

    out = parser.add_argument_group("Output")
    out.add_argument("--output", default=None, metavar="DIR",
                     help=(
                         "Output root directory. Default: a directory named after "
                         "--complex-cif's basename, in the cwd."
                     ))
    out.add_argument("--vis-dir", default=None, metavar="DIR",
                     help=(
                         "Directory collecting WT.cif (the reference complex) plus one "
                         "CIF per mutant named after its mutation (e.g. I33A.cif), for "
                         "loading together in a viewer (PyMOL/ChimeraX). "
                         "Default: best_structures/ under --output."
                     ))

    args = parser.parse_args()

    # ---- Stage 0: load the validated reference complex ----
    complex_cif = Path(args.complex_cif)
    if not complex_cif.exists():
        sys.exit(f"Error: complex CIF not found -- {args.complex_cif}")

    complex_model = load_structure(complex_cif)
    complex_seqs = extract_all_sequences(complex_model)
    if "A" not in complex_seqs or "B" not in complex_seqs:
        sys.exit(
            f"Error: expected chains 'A' (binder) and 'B' (antigen) in {args.complex_cif}, "
            f"found: {list(complex_seqs.keys())}"
        )
    wt_seq = complex_seqs["A"]
    antigen_seq = complex_seqs["B"]
    wt_name = complex_cif.stem

    print(f"Reference complex: {complex_cif.name}")
    print(f"  Binder (chain A):  {wt_name} ({len(wt_seq)} aa)")
    print(f"  Antigen (chain B): {len(antigen_seq)} aa\n")

    print("Stage 0: ANARCI (IMGT) numbering...")
    numbering = number_vhh(wt_seq)
    regions = assign_cdr_regions(numbering)
    cdrs_to_scan = {f"CDR{n}" for n in args.cdrs}
    cdr_counts = {c: sum(1 for r in regions.values() if r == c) for c in ["CDR1", "CDR2", "CDR3"]}
    print(f"  CDR lengths: CDR1={cdr_counts['CDR1']}  CDR2={cdr_counts['CDR2']}  "
          f"CDR3={cdr_counts['CDR3']}  (scanning: {', '.join(sorted(cdrs_to_scan))})\n")

    # ---- Output directories ----
    out_root = Path(args.output) if args.output else Path.cwd() / make_safe_name(wt_name)
    dock_dir = out_root / "docking"
    log_dir  = out_root / "logs"
    vis_dir  = Path(args.vis_dir) if args.vis_dir else out_root / "best_structures"
    for d in (dock_dir, log_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Reference complex always goes into the visualisation directory unconditionally --
    # it's already validated, so there's no confidence gate to apply here.
    ref_dest = copy_structure(complex_cif, vis_dir, "WT.cif")
    print(f"Reference complex copied to {ref_dest}\n")

    # ---- Stage 1: interface residues, read directly off the validated complex ----
    interface_positions = interface_positions_from_complex(complex_cif)
    if not interface_positions:
        sys.exit(
            "Error: no binder interface residues detected in --complex-cif "
            "(check that chain A/B are correctly labelled and in contact)."
        )
    print(f"Reference interface residues (binder, 1-based): {sorted(interface_positions)}")
    ref_metrics = reference_metrics(complex_cif)
    print(f"Reference geometry: bsa_A2={ref_metrics['bsa_A2']}  "
          f"n_clashes={ref_metrics['n_clashes']}\n")

    # ---- Stage 2: generate round-1 (Ala/Gly) mutants across the whole CDR panel ----
    mutants = generate_ala_scan_mutants(wt_seq, regions, cdrs_to_scan)
    if not mutants:
        sys.exit(
            "No residues found to scan -- --cdrs excludes every CDR loop, or ANARCI "
            "found no CDR residues in this sequence."
        )
    print(f"Stage 2: {len(mutants)} candidate mutation(s) to scan: "
          f"{', '.join(m[0] for m in mutants)}\n")

    # ---- Stage 3: dock each round-1 mutant, templated on the reference complex ----
    results = run_mutant_panel(
        mutants, wt_seq, wt_name, antigen_seq, complex_cif,
        dock_dir, log_dir, vis_dir, ref_metrics, args, round_label="round1",
        wt_binding_score=args.wt_binding_score,
    )

    # ---- Stage 5 (optional): classify positions, run enhancement panel ----
    if args.optimize:
        labels = classify_positions(results, interface_positions, args.hotspot_drop_fraction)
        n_hotspot = sum(1 for l in labels.values() if l == "hotspot")
        n_tolerant = sum(1 for l in labels.values() if l == "tolerant_contact")
        n_cold = sum(1 for l in labels.values() if l == "cold_spot")
        print(f"\nStage 5: classified {len(labels)} position(s) -- "
              f"{n_hotspot} hotspot (skipped), {n_tolerant} tolerant_contact, "
              f"{n_cold} cold_spot (both eligible for enhancement)\n")

        enhancement_mutants = generate_enhancement_mutants(wt_seq, regions, labels)
        if enhancement_mutants:
            print(f"Stage 5: {len(enhancement_mutants)} enhancement mutation(s) to try "
                  f"({', '.join(ENHANCEMENT_PANEL)} at each qualifying position): "
                  f"{', '.join(m[0] for m in enhancement_mutants)}\n")
            round2_results = run_mutant_panel(
                enhancement_mutants, wt_seq, wt_name, antigen_seq, complex_cif,
                dock_dir, log_dir, vis_dir, ref_metrics, args, round_label="round2",
                wt_binding_score=args.wt_binding_score,
            )
            results += round2_results
        else:
            print("Stage 5: no tolerant_contact or cold_spot positions found -- "
                  "every scanned position was classified as a hotspot.\n")

    # ---- Stage 4: report ----
    have_delta = args.wt_binding_score is not None
    out_df = pd.DataFrame(results)
    if not out_df.empty:
        if have_delta:
            # Ascending: biggest drops (hotspots) surface first, matching alanine-scan convention.
            out_df = out_df.sort_values("binding_score_delta", ascending=True, na_position="last")
        else:
            out_df = out_df.sort_values("mean_binding_score", ascending=False, na_position="last")
    out_csv = out_root / "mutation_candidates.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"\n{'='*70}")
    print(f"Done. {len(results)} mutant(s) scored.")
    print(f"Reference: bsa_A2={ref_metrics['bsa_A2']}  n_clashes={ref_metrics['n_clashes']}")
    if not out_df.empty:
        print(f"\nTop candidates by {'binding_score_delta' if have_delta else 'mean_binding_score'}:")
        cols = ["Round", "Variant", "CDR", "Position_label", "mean_binding_score",
                "binding_score_delta", "mean_iptm", "total_clashes"]
        print(out_df[cols].head(10).to_string(index=False))
    print(f"\nFull results: {out_csv}")
    print(f"Structures for visualisation: {vis_dir}")


if __name__ == "__main__":
    main()
