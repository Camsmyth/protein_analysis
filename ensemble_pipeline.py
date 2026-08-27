#!/usr/bin/env python3
"""
VHH–antigen docking pipeline.

End-to-end pipeline for confident VHH–antigen complex prediction from enriched
phage display cluster representatives. Runs three stages automatically:

  Stage 1  NanobodyBuilder2 (ImmuneBuilder) — generates 4 model-diverse VHH
           structures from independently pre-trained models. The rank-0 (best)
           structure is OpenMM-relaxed and used as the rigid-body template.
  Stage 2  Rigid-body Boltz2 docking — runs a two-chain (VHH + antigen) complex
           prediction with the rank-0 VHH as a fixed structural template for
           chain A and (optionally) the antigen as a template for chain B.
           Chain A never queries the MSA server — it always runs in
           single-sequence mode (msa: empty), since its conformation is
           already fixed by the NanobodyBuilder2 template and nanobody CDR
           loops carry little useful coevolutionary signal. --no-msa-server
           only affects chain B (the antigen). --recycling-steps recycling
           iterations (default: 3); --num-models diffusion samples (default: 5).
  Stage 3  Convergence scoring — epitope overlap + pose RMSD across diffusion
           samples. convergence_rank = epitope_overlap × mean_binding_score.

Input CSV columns (configurable via flags):
  Cluster              — cluster representative name  (--names)
  Protein_Sequence_R2  — VHH amino acid sequence       (--sequences)
  Log2_Enrichment      — panning enrichment score       (--enrichment)

Primary output: ensemble_binding_scores.csv
  One row per VHH sorted by convergence_rank = epitope_overlap * mean_binding_score.
  High convergence_rank + high Log2_Enrichment = strong wet-lab ordering candidate.

Usage:
    python ensemble_pipeline.py enriched_clusters.csv \\
        --antigen antigens/hCD7_alphafold.pdb \\
        --names Cluster --sequences Protein_Sequence_R2 \\
        --use-template --num-models 5

Key flags:
    --antigen PATH            Antigen structure file (.pdb or .cif) [required]
    --antigen-chain ID        Single chain from antigen (default: first)
    --antigen-chains ID ...   Multiple chains merged into one
    --use-template            Provide antigen as Boltz2 structural template
    --names COLUMN            Sample name column (default: Cluster)
    --sequences COLUMN        Sequence column (default: Protein_Sequence_R2)
    --enrichment COLUMN       Log2 enrichment column carried to output (default: Log2_Enrichment)
    --num-models N            Diffusion samples for rigid-body docking (default: 5)
    --max-parallel-samples N  GPU memory control for docking
    --no-msa-server           Disable ColabFold MSA server for the antigen (chain A/VHH
                              never uses it — see Stage 2 above)
    --output DIR              Output root directory
    --accelerator gpu/cpu/tpu

Outputs (under --output):
    vhh_structures/{name}/        NanobodyBuilder2 rank-0 PDB + CIF
    docking/{name}/               Boltz2 rigid-body docking results (all diffusion samples)
    ensemble_binding_scores.csv   Aggregated convergence scores (one row per VHH)
    ensemble_per_model.csv        Full per-docking-model detail
    logs/                         Boltz2 stdout/stderr logs
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser, SASA, Superimposer
from Bio.SeqUtils import seq1
from tqdm import tqdm

BOLTZ_BIN = Path(sys.executable).parent / "boltz"


# ---------------------------------------------------------------------------
# Structure utilities
# ---------------------------------------------------------------------------

def load_structure(path: Path):
    """Parse a PDB or mmCIF file and return the first model."""
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        sys.exit(f"Error: unsupported structure format '{suffix}'. Use .pdb or .cif.")
    structure = parser.get_structure(path.stem, str(path))
    return next(iter(structure))


def extract_all_sequences(model) -> dict[str, str]:
    """Return {chain_id: sequence} for all chains in a model."""
    seqs = {}
    for chain in model.get_chains():
        seq = "".join(
            seq1(r.resname)
            for r in chain.get_residues()
            if r.id[0] == " "
        )
        if seq:
            seqs[chain.id] = seq
    return seqs


# ---------------------------------------------------------------------------
# Antigen CIF preparation
# ---------------------------------------------------------------------------

def prepare_antigen_cif(antigen_path: Path) -> Path:
    """
    Return a CIF path for the antigen, converting from PDB if needed.

    PDB files without SEQRES records lack entity_poly_seq after gemmi conversion,
    which causes Boltz2's parse_polymer to fail with IndexError. This function
    detects that case and fills the entity sequence from ATOM records before
    writing the CIF. The result is saved alongside the original file and reused
    on subsequent runs.
    """
    if antigen_path.suffix.lower() in (".cif", ".mmcif"):
        return antigen_path

    cif_path = antigen_path.with_suffix(".cif")
    if cif_path.exists():
        return cif_path

    import gemmi  # only needed for PDB→CIF conversion

    st = gemmi.read_structure(str(antigen_path))
    st.setup_entities()

    for entity in st.entities:
        if entity.entity_type == gemmi.EntityType.Polymer and not entity.full_sequence:
            for chain in st[0]:
                polymer = chain.get_polymer()
                if not polymer:
                    continue
                poly_subchains = {res.subchain for res in polymer}
                if poly_subchains & set(entity.subchains):
                    entity.full_sequence = [res.name for res in polymer]
                    break

    subchain_counts: dict = {}
    subchain_renaming: dict = {}
    for chain in st[0]:
        subchain_counts[chain.name] = 0
        for res in chain:
            if res.subchain not in subchain_renaming:
                subchain_renaming[res.subchain] = (
                    chain.name + str(subchain_counts[chain.name] + 1)
                )
                subchain_counts[chain.name] += 1
            res.subchain = subchain_renaming[res.subchain]
    for entity in st.entities:
        entity.subchains = [subchain_renaming[s] for s in entity.subchains]

    doc = st.make_mmcif_document()
    doc.write_file(str(cif_path))
    print(f"  Converted {antigen_path.name} → {cif_path.name} (added SEQRES from ATOM records)")
    return cif_path


def merge_antigen_chains(antigen_path: Path, chain_ids: list[str]) -> tuple[Path, str]:
    """
    Merge multiple chains from an antigen structure into a single chain.
    Coordinates are preserved exactly; residues are renumbered sequentially.
    Returns (cif_path, merged_sequence).
    """
    label = "".join(chain_ids)
    pdb_path = antigen_path.parent / f"{antigen_path.stem}_chains{label}.pdb"

    model = load_structure(antigen_path)
    all_residues = []
    for cid in chain_ids:
        all_residues.extend(r for r in model[cid] if r.id[0] == " ")

    if not all_residues:
        sys.exit(f"Error: no ATOM residues found in chains {chain_ids} of {antigen_path.name}")

    lines = []
    atom_serial = 1
    for res_num, res in enumerate(all_residues, start=1):
        for atom in res.get_atoms():
            x, y, z = atom.coord
            lines.append(
                f"ATOM  {atom_serial:5d} {atom.name:<4s} {res.resname:<3s} A{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{atom.occupancy:6.2f}{atom.bfactor:6.2f}          "
                f"{atom.element:>2s}\n"
            )
            atom_serial += 1
    lines.append("END\n")
    pdb_path.write_text("".join(lines))

    merged_seq = "".join(seq1(r.resname) for r in all_residues)
    cif_path = prepare_antigen_cif(pdb_path)
    print(f"  Merged chains {chain_ids} → {pdb_path.name} ({len(all_residues)} residues)")
    return cif_path, merged_seq


# ---------------------------------------------------------------------------
# YAML generation
# ---------------------------------------------------------------------------


def write_complex_yaml(
    binder_seq: str,
    antigen_seq: str,
    path: Path,
    binder_structure: Path | None = None,
    antigen_structure: Path | None = None,
) -> None:
    """Write a two-chain complex YAML for Boltz2. Chain A = binder, Chain B = antigen."""
    lines = [
        "sequences:",
        "  - protein:",
        "      id: A",
        f'      sequence: "{binder_seq}"',
        # Chain A always has a NanobodyBuilder2 template fixing its conformation, and
        # nanobody CDR loops carry little useful coevolutionary signal — MSA and
        # template are additive in Boltz2, not exclusive, so skip the (novel,
        # proprietary) VHH MSA server call entirely rather than relying on
        # --no-msa-server.
        "      msa: empty",
        "  - protein:",
        "      id: B",
        f'      sequence: "{antigen_seq}"',
    ]
    templates = []
    if binder_structure:
        key = "cif" if binder_structure.suffix.lower() in (".cif", ".mmcif") else "pdb"
        templates += [f"  - {key}: {binder_structure.resolve()}", "    chain_id: [A]"]
    if antigen_structure:
        key = "cif" if antigen_structure.suffix.lower() in (".cif", ".mmcif") else "pdb"
        templates += [f"  - {key}: {antigen_structure.resolve()}", "    chain_id: [B]"]
    if templates:
        lines += ["templates:"] + templates
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Boltz runner
# ---------------------------------------------------------------------------

def run_boltz(
    input_dir: Path,
    out_dir: Path,
    accelerator: str,
    num_models: int,
    no_msa_server: bool,
    recycling_steps: int = 3,
    max_parallel_samples: int | None = None,
    log_path: Path | None = None,
) -> int:
    """Invoke the boltz CLI."""
    cmd = [
        str(BOLTZ_BIN), "predict", str(input_dir),
        "--out_dir", str(out_dir),
        "--accelerator", accelerator,
        "--diffusion_samples", str(num_models),
        "--recycling_steps", str(recycling_steps),
        "--model", "boltz2",
        "--write_full_pae",
        "--num_workers", "4",
        "--preprocessing-threads", "4",
    ]
    if max_parallel_samples is not None:
        cmd += ["--max_parallel_samples", str(max_parallel_samples)]
    if not no_msa_server:
        cmd.append("--use_msa_server")

    if log_path:
        tqdm.write(f"  → boltz2 log: {log_path}")
        with log_path.open("w") as lf:
            lf.write(f"CMD: {' '.join(cmd)}\n\n")
            lf.flush()
            result = subprocess.run(cmd, stdout=lf, stderr=lf)
        rc = result.returncode
        if rc != 0:
            lines = log_path.read_text().splitlines()
            tqdm.write(f"\n  Boltz2 failed (exit {rc}). Last lines from log:")
            for line in lines[-20:]:
                tqdm.write(f"    {line}")
        return rc
    else:
        tqdm.write(f"  Running: {' '.join(cmd)}")
        return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def parse_confidence_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {
        "confidence_score": data.get("confidence_score"),
        "ptm":              data.get("ptm"),
        "iptm":             data.get("iptm"),
        "protein_iptm":     data.get("protein_iptm"),
        "complex_plddt":    data.get("complex_plddt"),
        "complex_iplddt":   data.get("complex_iplddt"),
        "complex_pde":      data.get("complex_pde"),
        "complex_ipde":     data.get("complex_ipde"),
        "chain_iptm_A_B":   (
            data.get("pair_chains_iptm", {}).get("0", {}).get("1")
            or data.get("pair_chains_iptm", {}).get(0, {}).get(1)
        ),
    }


def parse_interface_pae(pae_path: Path, binder_len: int, antigen_len: int) -> dict:
    """
    Mean PAE across the binder–antigen interface from the full PAE matrix.
    Rows/cols 0..binder_len-1 = binder (A), binder_len..N-1 = antigen (B).
    """
    pae = np.load(pae_path)["pae"]
    b = binder_len
    pae_ab = pae[:b, b:]
    pae_ba = pae[b:, :b]
    return {
        "pae_interface": float(np.mean([pae_ab.mean(), pae_ba.mean()])),
        "pae_binder":    float(pae[:b, :b].mean()),
        "pae_antigen":   float(pae[b:, b:].mean()),
    }


def calculate_bsa(
    structure_path: Path,
    binder_chain: str = "A",
    antigen_chain: str = "B",
) -> float | None:
    """Buried surface area (Å²) = SASA(binder) + SASA(antigen) - SASA(complex)."""
    try:
        model = load_structure(structure_path)
        sr = SASA.ShrakeRupley()
        sr.compute(model, level="R")
        sasa_complex = sum(r.sasa for chain in model for r in chain if r.id[0] == " ")
        sr.compute(model[binder_chain], level="R")
        binder_sasa = sum(r.sasa for r in model[binder_chain] if r.id[0] == " ")
        sr.compute(model[antigen_chain], level="R")
        antigen_sasa = sum(r.sasa for r in model[antigen_chain] if r.id[0] == " ")
        return round(float((binder_sasa + antigen_sasa - sasa_complex) / 2), 2)
    except Exception as e:
        print(f"  Warning: BSA calculation failed — {e}", file=sys.stderr)
        return None


def count_interface_contacts(
    structure_path: Path,
    binder_chain: str = "A",
    antigen_chain: str = "B",
    threshold_a: float = 8.0,
) -> dict:
    """Count Cα–Cα residue pairs within threshold_a Å across the interface."""
    try:
        model = load_structure(structure_path)
        binder_res  = [r for r in model[binder_chain]  if r.id[0] == " " and "CA" in r]
        antigen_res = [r for r in model[antigen_chain] if r.id[0] == " " and "CA" in r]
        contacts = 0
        binder_iface:  dict[int, str] = {}
        antigen_iface: dict[int, str] = {}
        for br in binder_res:
            for ar in antigen_res:
                if br["CA"] - ar["CA"] <= threshold_a:
                    contacts += 1
                    binder_iface[br.id[1]]  = seq1(br.resname)
                    antigen_iface[ar.id[1]] = seq1(ar.resname)

        def _fmt(d: dict) -> str:
            residues = ",".join(f"{aa}{num}" for num, aa in sorted(d.items()))
            return f"{len(d)} ({residues})" if residues else "0"

        return {
            "interface_contacts":         contacts,
            "binder_interface_residues":  _fmt(binder_iface),
            "antigen_interface_residues": _fmt(antigen_iface),
        }
    except Exception as e:
        print(f"  Warning: contact calculation failed — {e}", file=sys.stderr)
        return {
            "interface_contacts": None,
            "binder_interface_residues": None,
            "antigen_interface_residues": None,
        }


def count_heavy_atom_clashes(
    structure_path: Path,
    binder_chain: str = "A",
    antigen_chain: str = "B",
    min_dist: float = 1.5,
) -> int:
    """
    Count cross-chain heavy-atom pairs closer than min_dist Å.
    Any value > 0 indicates a physically impossible overlap; used to flag
    implausible docking poses before they reach the report.
    """
    try:
        from Bio.PDB import NeighborSearch
        model = load_structure(structure_path)
        antigen_atoms = list(model[antigen_chain].get_atoms())
        binder_atoms  = list(model[binder_chain].get_atoms())
        ns = NeighborSearch(antigen_atoms)
        return sum(1 for ba in binder_atoms if ns.search(ba.coord, min_dist, level="A"))
    except Exception as e:
        print(f"  Warning: clash detection failed — {e}", file=sys.stderr)
        return 0


def binding_score(iptm: float | None, ptm: float | None) -> float | None:
    """0.8 * ipTM + 0.2 * pTM — AF2-multimer / BindCraft convention."""
    if iptm is not None and ptm is not None:
        return round(0.8 * iptm + 0.2 * ptm, 4)
    return None


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
        sys.exit(f"Error: column(s) not found: {missing}\nAvailable: {list(df.columns)}")
    return df


def clean_sequence(raw: object) -> str:
    return str(raw).strip().upper().replace(" ", "").replace("\n", "")


def make_safe_name(raw: object) -> str:
    return str(raw).strip().replace(" ", "_").replace("/", "_")


def default_output_dir(input_path: str) -> Path:
    return Path(input_path).parent / "ensemble_predictions"


# ---------------------------------------------------------------------------
# Stage 1 — NanobodyBuilder2 conformer ensemble
# ---------------------------------------------------------------------------

def run_immunebuilder_best(sequence: str, name: str, out_dir: Path) -> Path | None:
    """
    Run NanoBodyBuilder2, OpenMM-relax the rank-0 (best) model, and return its CIF path.

    NanobodyBuilder2 ranks its 4 independently pre-trained model outputs; rank-0 is
    the highest-confidence structure by its internal scoring. That structure is
    refined with OpenMM (backbone-restrained energy minimisation) then converted to
    CIF so Boltz2 can parse entity_poly_seq (NanobodyBuilder2 PDBs lack SEQRES).

    Returns None if ImmuneBuilder is not installed or prediction fails.
    """
    try:
        from ImmuneBuilder import NanoBodyBuilder2  # type: ignore
        from ImmuneBuilder.refine import refine      # type: ignore
    except ImportError:
        tqdm.write("  [Stage 1] ImmuneBuilder not installed — skipping.")
        return None
    try:
        builder = NanoBodyBuilder2()
        nanobody = builder.predict({"H": sequence})

        best_idx = nanobody.ranking[0]
        pdb_path = out_dir / f"{name}_best_model.pdb"
        nanobody.save_single_unrefined(str(pdb_path), index=best_idx)

        success = refine(str(pdb_path), str(pdb_path))
        if success:
            tqdm.write(f"  [Stage 1] Best model (rank-0) OpenMM-relaxed ✓ → {pdb_path.name}")
        else:
            tqdm.write(f"  [Stage 1] OpenMM refinement failed — using unrefined best model.")

        cif_path = prepare_antigen_cif(pdb_path)
        tqdm.write(f"  [Stage 1] Best model CIF → {cif_path.name}")
        return cif_path
    except Exception as e:
        tqdm.write(f"  [Stage 1] NanobodyBuilder2 failed for '{name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 2 — Rigid-body Boltz2 docking
# ---------------------------------------------------------------------------

def _score_model(
    pred_dir: Path,
    run_name: str,
    model_idx: int,
    binder_len: int,
    antigen_len: int,
    run_out: Path,
) -> dict:
    """Extract all metrics for one Boltz2 diffusion sample. Copies CIF to run_out."""
    conf_path = pred_dir / f"confidence_{run_name}_model_{model_idx}.json"
    pae_path  = pred_dir / f"pae_{run_name}_model_{model_idx}.npz"
    cif_path  = pred_dir / f"{run_name}_model_{model_idx}.cif"

    row: dict = {}

    if conf_path.exists():
        row.update(parse_confidence_json(conf_path))
    else:
        row.update({k: None for k in [
            "confidence_score", "ptm", "iptm", "protein_iptm",
            "complex_plddt", "complex_iplddt",
            "complex_pde", "complex_ipde", "chain_iptm_A_B",
        ]})

    if pae_path.exists():
        row.update(parse_interface_pae(pae_path, binder_len, antigen_len))
    else:
        row.update({k: None for k in ["pae_interface", "pae_binder", "pae_antigen"]})

    dest_num = model_idx + 1
    if cif_path.exists():
        row["bsa_A2"] = calculate_bsa(cif_path)
        row.update(count_interface_contacts(cif_path))
        row["n_clashes"] = count_heavy_atom_clashes(cif_path)
        shutil.copy2(cif_path, run_out / f"{run_name}_model_{dest_num}.cif")
    else:
        row.update({k: None for k in [
            "bsa_A2", "interface_contacts", "n_clashes",
            "binder_interface_residues", "antigen_interface_residues",
        ]})

    row["binding_score"] = binding_score(row.get("iptm"), row.get("ptm"))
    return row


def dock_vhh(
    name: str,
    best_model_cif: Path,
    binder_seq: str,
    antigen_seq: str,
    antigen_cif: Path,
    dock_root: Path,
    accelerator: str,
    num_models: int,
    no_msa_server: bool,
    use_template: bool,
    max_parallel_samples: int | None,
    log_dir: Path,
    recycling_steps: int = 3,
) -> list[dict]:
    """
    Rigid-body Boltz2 docking for one VHH against the antigen.

    The rank-0 OpenMM-relaxed VHH structure (best_model_cif) is provided as a
    structural template for chain A, fixing its conformation during diffusion.
    If use_template is True the antigen CIF is also templated (chain B), making
    the docking fully rigid-body. num_models diffusion samples are run in a
    single Boltz2 call with the given recycling_steps.

    Returns a list of per-model metric dicts (one per diffusion sample).
    """
    run_name = name
    run_out = dock_root / run_name
    run_out.mkdir(parents=True, exist_ok=True)

    binder_len  = len(binder_seq)
    antigen_len = len(antigen_seq)

    log_path = log_dir / f"{run_name}.log"
    with tempfile.TemporaryDirectory() as yaml_tmp:
        yaml_path = Path(yaml_tmp) / f"{run_name}.yaml"
        write_complex_yaml(
            binder_seq, antigen_seq, yaml_path,
            binder_structure=best_model_cif,
            antigen_structure=antigen_cif if use_template else None,
        )
        rc = run_boltz(
            Path(yaml_tmp), run_out, accelerator, num_models, no_msa_server,
            recycling_steps=recycling_steps,
            max_parallel_samples=max_parallel_samples,
            log_path=log_path,
        )

    if rc != 0:
        tqdm.write(f"  [Stage 2] Boltz2 failed for '{name}'.")
        return []

    pred_dirs = list(run_out.glob(f"*/predictions/{run_name}"))
    if not pred_dirs:
        tqdm.write(f"  [Stage 2] No Boltz2 prediction directory found for '{name}'.")
        return []
    pred_dir = pred_dirs[0]

    rows: list[dict] = []
    for model_idx in range(num_models):
        row = _score_model(pred_dir, run_name, model_idx, binder_len, antigen_len, run_out)
        n_cl = row.get("n_clashes") or 0
        clash_tag = "✓" if n_cl == 0 else f"{n_cl} clash(es)"
        tqdm.write(
            f"  [Stage 2] model {model_idx + 1}/{num_models}: "
            f"binding_score={row.get('binding_score')}  clashes={clash_tag}"
        )
        row_out = {"Sample": name, "conformer": 1, "Model": model_idx + 1}
        row_out.update(row)
        rows.append(row_out)

    return rows


# ---------------------------------------------------------------------------
# Stage 5 — Convergence scoring
# ---------------------------------------------------------------------------

def _parse_antigen_residue_set(s: object) -> set[int]:
    """Parse '13 (A2,Q3,...)' → {2, 3, ...} antigen residue number set."""
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


def compute_epitope_overlap(all_rows: list[dict]) -> float | None:
    """
    Fraction of antigen interface residues shared across ≥50% of docking results.

    A high value means multiple CDR3 conformers independently converge on the same
    antigen epitope — a strong signal that the binding mode is robust, not an
    artefact of a single loop geometry.
    """
    from collections import Counter
    valid = [r for r in all_rows if r.get("antigen_interface_residues")]
    if not valid:
        return None
    counter: Counter = Counter()
    for r in valid:
        counter.update(_parse_antigen_residue_set(r["antigen_interface_residues"]))
    consensus = {res for res, cnt in counter.items() if cnt >= max(1, len(valid) / 2)}
    all_res = set(counter.keys())
    return round(len(consensus) / len(all_res), 4) if all_res else None


def compute_pose_convergence_rmsd(
    dock_root: Path,
    name: str,
    n_models: int,
) -> float | None:
    """
    Superimpose all diffusion-sample complexes onto antigen chain B of model 1,
    then report mean Cα RMSD of binder chain A across samples.
    Low RMSD = diffusion samples converge on the same pose.
    """
    run_dir = dock_root / name
    cif_paths = []
    for m in range(1, n_models + 1):
        p = run_dir / f"{name}_model_{m}.cif"
        if p.exists():
            cif_paths.append(p)

    if len(cif_paths) < 2:
        return None

    try:
        models = [load_structure(p) for p in cif_paths]
        ref = models[0]
        ref_ant_ca = [r["CA"] for r in ref["B"] if r.id[0] == " " and "CA" in r]
        ref_bin_ca = [r["CA"] for r in ref["A"] if r.id[0] == " " and "CA" in r]

        sup = Superimposer()
        rmsds = []
        for other in models[1:]:
            other_ant_ca = [r["CA"] for r in other["B"] if r.id[0] == " " and "CA" in r]
            other_bin_ca = [r["CA"] for r in other["A"] if r.id[0] == " " and "CA" in r]
            n_ant = min(len(ref_ant_ca), len(other_ant_ca))
            n_bin = min(len(ref_bin_ca), len(other_bin_ca))
            if n_ant < 3 or n_bin < 3:
                continue
            sup.set_atoms(ref_ant_ca[:n_ant], other_ant_ca[:n_ant])
            sup.apply(list(other.get_atoms()))
            diff = np.array([
                a1.coord - a2.coord
                for a1, a2 in zip(ref_bin_ca[:n_bin], other_bin_ca[:n_bin])
            ])
            rmsds.append(float(np.sqrt((diff ** 2).sum(axis=1).mean())))

        return round(float(np.mean(rmsds)), 3) if rmsds else None
    except Exception as e:
        tqdm.write(f"  [Stage 3] Pose convergence RMSD failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

# Thresholds used for tiering and inline quality flags
_BSA_HIGH    = 1000  # Å² — above this, verify interface specificity
_BSA_GOOD    = 600   # Å² — lower bound of good range (600–1000)
_BSA_OK      = 600   # Å² — below this is weak
_PAE_GOOD    = 10.0  # Å  — low cross-chain PAE = high confidence
_PAE_OK      = 15.0
_IPTM_READY  = 0.65
_BS_READY    = 0.65
_OV_READY    = 0.50
_RMSD_READY  = 5.0
_IPTM_UNCERT = 0.45
_BS_UNCERT   = 0.45
_OV_UNCERT   = 0.20


def _tier(row: pd.Series, clashes: int) -> str:
    """
    Three-tier classification. Clashes disqualify from READY regardless of scores.

    READY     — strong binding, consistent epitope, no clashes → interface analysis
    REVIEW    — moderate scores or partial convergence or minor clashes
    UNCERTAIN — low binding, divergent poses, or severe clashes
    """
    bs   = row.get("mean_binding_score")
    ip   = row.get("mean_iptm")
    ov   = row.get("epitope_overlap_fraction")
    pae  = row.get("mean_pae_interface")
    rmsd = row.get("pose_convergence_rmsd")
    bsa  = row.get("mean_bsa_A2")

    if (
        bs   is not None and bs   >= _BS_READY and
        ip   is not None and ip   >= _IPTM_READY and
        ov   is not None and ov   >= _OV_READY and
        (pae  is None or pae  <= _PAE_OK) and
        (rmsd is None or rmsd <= _RMSD_READY) and
        clashes == 0
    ):
        return "READY"
    # BSA deliberately excluded from READY gate: high BSA can indicate non-specific
    # burial; low BSA does not alone disqualify — epitope convergence matters more.

    if (
        (bs  is not None and bs  < _BS_UNCERT) or
        (ip  is not None and ip  < _IPTM_UNCERT) or
        (ov  is not None and ov  < _OV_UNCERT) or
        clashes > 5
    ):
        return "UNCERTAIN"

    return "REVIEW"


def _liability_flags(liabilities: object) -> list[str]:
    """Return list of compact liability labels from a semicolon-separated string."""
    s = str(liabilities or "").strip()
    if not s or s.lower() == "nan":
        return []
    abbrevs = {
        "Free Cys":          "FreeCys ⚠",
        "Oxidation (Met)":   "OxMet",
        "Isomerization":     "Isomer",
        "Deamidation":       "Deamid",
        "Aggregation prone": "Aggr ⚠",
    }
    return [abbrevs.get(p.strip(), p.strip()) for p in s.split(";") if p.strip()]


def _bsa_label(bsa: float | None) -> str:
    if bsa is None:
        return ""
    if bsa > _BSA_HIGH:
        return " ⚠ verify interface (>1000 Å²)"
    if bsa >= _BSA_GOOD:
        return " ✓ good (600–1000 Å²)"
    return " ✗ weak (<600 Å²)"


def _pae_label(pae: float | None) -> str:
    if pae is None:
        return ""
    if pae <= _PAE_GOOD:
        return " ✓ confident"
    if pae <= _PAE_OK:
        return " ~ acceptable"
    return " ✗ uncertain"


def _clash_label(n: int) -> str:
    if n == 0:
        return "0 ✓"
    if n <= 3:
        return f"{n} ~ (minor)"
    return f"{n} ✗ (severe)"


def _find_best_cif(out_dir: Path, name: str, per_model_df: pd.DataFrame) -> str:
    """Return path to the CIF with the highest binding_score for this VHH."""
    dock_root = out_dir / "docking"
    run_dir = dock_root / name
    subset = per_model_df[per_model_df["Sample"] == name].copy()
    if not subset.empty and "binding_score" in subset.columns:
        subset = subset.sort_values("binding_score", ascending=False)
        for _, model_row in subset.iterrows():
            m = int(model_row.get("Model", 1))
            p = run_dir / f"{name}_model_{m}.cif"
            if p.exists():
                return str(p)
    # Fallback: scan for any existing CIF
    for m in range(1, 20):
        p = run_dir / f"{name}_model_{m}.cif"
        if p.exists():
            return str(p)
    return f"{run_dir}/ (not yet generated)"


def print_final_report(
    results_df: pd.DataFrame,
    input_df: pd.DataFrame,
    per_model_df: pd.DataFrame,
    out_dir: Path,
    names_col: str,
) -> None:
    """
    Print a structured, human-readable tier report and save report.txt.

    Shows only decision-relevant information: CDR sequences, key biophysical
    properties, developability liabilities, enrichment signal, and a detailed
    structural binding assessment (binding score, ipTM, PAE, BSA, interface
    contacts, clash count, epitope convergence).
    """
    input_copy = input_df.copy()
    input_copy["_safe_name"] = input_copy[names_col].apply(make_safe_name)
    merged = results_df.merge(input_copy, left_on="Cluster", right_on="_safe_name", how="left")

    # Aggregate clash and contact counts from per-model data per VHH
    clash_agg: dict[str, int] = {}
    contact_agg: dict[str, float] = {}
    plddt_agg: dict[str, float] = {}
    if not per_model_df.empty:
        for vhh_name, grp in per_model_df.groupby("Sample"):
            clashes = grp["interface_contacts"].count()  # use as proxy check
            # n_clashes column not in summary — read from per_model if present
            if "n_clashes" in grp.columns:
                clash_agg[vhh_name] = int(grp["n_clashes"].fillna(0).sum())
            else:
                clash_agg[vhh_name] = 0
            if "interface_contacts" in grp.columns:
                vals = grp["interface_contacts"].dropna()
                contact_agg[vhh_name] = float(vals.mean()) if not vals.empty else 0.0
            if "complex_plddt" in grp.columns:
                vals = grp["complex_plddt"].dropna()
                plddt_agg[vhh_name] = float(vals.mean()) if not vals.empty else 0.0

    merged["_clashes"]  = merged["Cluster"].map(lambda n: clash_agg.get(n, 0))
    merged["_contacts"] = merged["Cluster"].map(lambda n: contact_agg.get(n, 0.0))
    merged["_plddt"]    = merged["Cluster"].map(lambda n: plddt_agg.get(n, 0.0))
    merged["_tier"]     = merged.apply(
        lambda r: _tier(r, int(r["_clashes"])), axis=1
    )

    tier_order = {"READY": 0, "REVIEW": 1, "UNCERTAIN": 2}
    merged["_tier_order"] = merged["_tier"].map(tier_order)
    merged = merged.sort_values(
        ["_tier_order", "convergence_rank"], ascending=[True, False], na_position="last"
    )

    ready     = merged[merged["_tier"] == "READY"]
    review    = merged[merged["_tier"] == "REVIEW"]
    uncertain = merged[merged["_tier"] == "UNCERTAIN"]

    lines: list[str] = []

    def _w(*args: str) -> None:
        lines.append(" ".join(str(a) for a in args) if args else "")

    def _fmt(val: object, fmt: str = ".3f", unit: str = "") -> str:
        if val is None or (isinstance(val, float) and np.isnan(float(val) if val == val else float("nan"))):
            return "—"
        try:
            return f"{float(val):{fmt}}{unit}"
        except (TypeError, ValueError):
            return str(val)

    W = 78
    _w("=" * W)
    _w(" VHH–ANTIGEN RIGID-BODY DOCKING  —  CANDIDATE REPORT")
    _w("=" * W)
    _w(f" {len(results_df)} VHH(s) evaluated   "
       f"READY: {len(ready)}   REVIEW: {len(review)}   UNCERTAIN: {len(uncertain)}")
    _w()
    _w("  READY     ipTM ≥ 0.65, binding_score ≥ 0.65, epitope_overlap ≥ 50%,")
    _w("            PAE ≤ 15 Å, pose RMSD ≤ 5 Å, 0 clashes")
    _w("            BSA flagged ⚠ if > 1000 Å² (may reflect non-specific burial)")
    _w("  REVIEW    borderline scores or minor structural issues — inspect manually")
    _w("  UNCERTAIN low confidence or severe clashes — do not order")
    _w()

    def _entry(r: pd.Series) -> None:
        name = r["Cluster"]

        # --- Selection / enrichment ---
        enrich   = _fmt(r.get("Log2_Enrichment"), ".2f")
        fdr      = _fmt(r.get("Neg_log10_FDR"), ".1f")
        count_r2 = r.get("Count_R2")
        count_r2_s = f"{int(count_r2)}" if pd.notna(count_r2) else "—"
        uniq     = r.get("Unique_Sequences_R2")
        uniq_s   = f"{int(uniq)}" if pd.notna(uniq) else "—"
        qflag    = str(r.get("Quality_Flag", "") or "")
        qflag_s  = f"  [{qflag}]" if qflag.lower() not in ("nan", "", "none") else ""

        # --- CDR sequences ---
        cdr1     = str(r.get("CDR1_R2") or "—")
        cdr2     = str(r.get("CDR2_R2") or "—")
        cdr3     = str(r.get("Representative_CDR3_R2") or r.get("CDR3") or "—")
        cdr3_len = r.get("CDR3_Length_R2")
        cdr3_len_s = f"{int(cdr3_len)} aa" if pd.notna(cdr3_len) else ""

        # --- Biophysical / developability ---
        mw      = _fmt(r.get("MW_kDa_R2"), ".1f", " kDa")
        pi      = _fmt(r.get("pI_R2"), ".2f")
        chg     = _fmt(r.get("Charge_pH74_R2"), "+.1f")
        gravy   = _fmt(r.get("GRAVY_R2"), "+.3f")
        liabs   = _liability_flags(r.get("Liabilities_R2"))
        liab_s  = "  ".join(liabs) if liabs else "none"

        # --- Confidence scores ---
        bs      = _fmt(r.get("mean_binding_score"), ".3f")
        bsd     = _fmt(r.get("binding_score_std"), ".3f")
        bsb     = _fmt(r.get("best_binding_score"), ".3f")
        ip      = _fmt(r.get("mean_iptm"), ".3f")
        bip     = _fmt(r.get("best_iptm"), ".3f")
        plddt   = _fmt(r.get("_plddt") if r.get("_plddt") else None, ".1f")
        rank    = _fmt(r.get("convergence_rank"), ".4f")

        # --- Interface quality ---
        pae_v   = r.get("mean_pae_interface")
        pae     = _fmt(pae_v, ".1f", " Å") + _pae_label(pae_v)
        bsa_v   = r.get("mean_bsa_A2")
        bsa     = _fmt(bsa_v, ".0f", " Å²") + _bsa_label(bsa_v)
        contacts_v = r.get("_contacts")
        contacts = _fmt(contacts_v, ".0f", " residue pairs") if contacts_v else "—"
        clashes_v = int(r.get("_clashes", 0))
        clashes  = _clash_label(clashes_v)

        # --- Convergence ---
        ov_v    = r.get("epitope_overlap_fraction")
        ov      = _fmt(ov_v, ".0%")
        ov_note = (" ✓ consistent" if ov_v is not None and ov_v >= _OV_READY
                   else (" ~ partial" if ov_v is not None and ov_v >= 0.30
                         else " ✗ divergent"))
        rmsd_v  = r.get("pose_convergence_rmsd")
        rmsd    = _fmt(rmsd_v, ".2f", " Å")
        rmsd_note = (" ✓" if rmsd_v is not None and rmsd_v <= _RMSD_READY else
                     (" ~" if rmsd_v is not None and rmsd_v <= 8.0 else " ✗"))
        nconf   = r.get("n_conformers_kept", "—")
        ndock   = r.get("n_docking_runs", "—")

        _w()
        _w(f"  ┌─ {name}{qflag_s}")
        _w(f"  │  CDRs        CDR1: {cdr1}   CDR2: {cdr2}")
        _w(f"  │              CDR3: {cdr3} ({cdr3_len_s})")
        _w(f"  │  Biophysical MW: {mw}   pI: {pi}   charge(pH 7.4): {chg}   GRAVY: {gravy}")
        if liabs:
            _w(f"  │  Liabilities {liab_s}")
        _w(f"  │  Enrichment  log2FC: {enrich}   −log10(FDR): {fdr}   "
           f"R2 count: {count_r2_s}   cluster size: {uniq_s} seqs")
        _w(f"  │")
        _w(f"  │  ── Binding confidence ──────────────────────────────────────")
        _w(f"  │  binding_score  {bs} ± {bsd}  (best: {bsb})")
        _w(f"  │  ipTM           {ip}  (best: {bip})")
        if plddt != "—":
            _w(f"  │  complex pLDDT {plddt}")
        _w(f"  │")
        _w(f"  │  ── Interface quality ───────────────────────────────────────")
        _w(f"  │  PAE (interface) {pae}")
        _w(f"  │  BSA             {bsa}")
        _w(f"  │  Contacts        {contacts}")
        _w(f"  │  Clashes         {clashes}")
        _w(f"  │")
        _w(f"  │  ── Pose convergence ({nconf} conformers, {ndock} docking runs) ──────")
        _w(f"  │  Epitope overlap  {ov}{ov_note}")
        _w(f"  │  Pose RMSD        {rmsd}{rmsd_note}")
        _w(f"  │  Convergence rank {rank}")

        tier = r["_tier"]
        if tier == "READY":
            cif = _find_best_cif(out_dir, name, per_model_df)
            _w(f"  │")
            _w(f"  │  ★ READY FOR INTERFACE ANALYSIS")
            _w(f"  │    Best complex: {cif}")
        elif tier == "REVIEW":
            concerns = []
            bs_val = r.get("mean_binding_score")
            ip_val = r.get("mean_iptm")
            if bs_val is not None and bs_val < _BS_READY:
                concerns.append(f"binding_score {bs_val:.3f} < {_BS_READY}")
            if ip_val is not None and ip_val < _IPTM_READY:
                concerns.append(f"ipTM {ip_val:.3f} < {_IPTM_READY}")
            if ov_v is not None and ov_v < _OV_READY:
                concerns.append(f"epitope_overlap {ov_v:.0%} < {_OV_READY:.0%}")
            if bsa_v is not None and bsa_v > _BSA_HIGH:
                concerns.append(f"BSA {bsa_v:.0f} Å² > 1000 — verify interface specificity")
            elif bsa_v is not None and bsa_v < _BSA_GOOD:
                concerns.append(f"BSA {bsa_v:.0f} Å² < 600 — weak burial")
            if rmsd_v is not None and rmsd_v > _RMSD_READY:
                concerns.append(f"pose RMSD {rmsd_v:.2f} Å > {_RMSD_READY} Å")
            if clashes_v > 0:
                concerns.append(f"{clashes_v} clash(es)")
            if liabs:
                concerns.append("developability liabilities present")
            if concerns:
                _w(f"  │")
                _w(f"  │  ⚑  Concerns: {';  '.join(concerns)}")
        else:  # UNCERTAIN
            reasons = []
            bs_val = r.get("mean_binding_score")
            ip_val = r.get("mean_iptm")
            if bs_val is not None and bs_val < _BS_UNCERT:
                reasons.append(f"binding_score {bs_val:.3f} below threshold")
            if ip_val is not None and ip_val < _IPTM_UNCERT:
                reasons.append(f"ipTM {ip_val:.3f} below threshold")
            if ov_v is not None and ov_v < _OV_UNCERT:
                reasons.append(f"epitope_overlap {ov_v:.0%} — poses divergent")
            if clashes_v > 5:
                reasons.append(f"{clashes_v} steric clashes")
            if reasons:
                _w(f"  │")
                _w(f"  │  ✗  Reasons: {';  '.join(reasons)}")

        _w(f"  └{'─' * (W - 3)}")

    def _block(subset: pd.DataFrame, header: str) -> None:
        if subset.empty:
            return
        n = len(subset)
        _w(f"{'━' * W}")
        _w(f"  {header}  ({n} VHH{'s' if n != 1 else ''})")
        _w(f"{'━' * W}")
        for _, r in subset.iterrows():
            _entry(r)
        _w()

    _block(ready,     "★  READY — proceed to interface analysis")
    _block(review,    "⚑  REVIEW — inspect before ordering")
    _block(uncertain, "✗  UNCERTAIN — do not order")

    _w("=" * W)
    _w(f"  Summary CSV : {out_dir / 'ensemble_binding_scores.csv'}")
    _w(f"  Per-model   : {out_dir / 'ensemble_per_model.csv'}")
    _w(f"  Report      : {out_dir / 'report.txt'}")
    _w(f"  Complexes   : {out_dir / 'docking'}/")
    _w("=" * W)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    (out_dir / "report.txt").write_text(report_text + "\n")


# ---------------------------------------------------------------------------
# Output column order
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "Cluster", "sequence", "Log2_Enrichment",
    "n_conformers_kept", "n_docking_runs",
    "mean_binding_score", "best_binding_score", "binding_score_std",
    "mean_iptm", "best_iptm",
    "mean_pae_interface", "mean_bsa_A2",
    "epitope_overlap_fraction", "pose_convergence_rmsd",
    "convergence_rank",
]

PER_MODEL_COLUMNS = [
    "Sample", "conformer", "Model",
    "binding_score", "confidence_score", "iptm", "protein_iptm", "ptm",
    "complex_plddt", "complex_iplddt",
    "pae_interface", "pae_binder", "pae_antigen",
    "complex_pde", "complex_ipde",
    "bsa_A2", "interface_contacts", "n_clashes",
    "binder_interface_residues", "antigen_interface_residues",
    "chain_iptm_A_B",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "VHH–antigen docking pipeline.\n\n"
            "Runs NanobodyBuilder2 to generate 4 VHH structures, selects the rank-0 "
            "(best) model after OpenMM relaxation, then runs rigid-body Boltz2 complex "
            "prediction with that structure as the VHH template "
            "(--recycling-steps, default: 3). "
            "Scores binding confidence by epitope overlap and pose RMSD across diffusion "
            "samples."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Input/output ----
    io = parser.add_argument_group("Input / output")
    io.add_argument("input",
                    help="CSV or Excel file containing VHH cluster representatives.")
    io.add_argument("--output", default=None, metavar="DIR",
                    help=(
                        "Root output directory. All stages write here. "
                        "Default: ensemble_predictions/ in the same directory as the input file."
                    ))
    io.add_argument("--names", default="Cluster", metavar="COLUMN",
                    help="Column containing the VHH name / cluster ID. Default: Cluster")
    io.add_argument("--sequences", default="Protein_Sequence_R2", metavar="COLUMN",
                    help="Column containing VHH amino acid sequences. Default: Protein_Sequence_R2")
    io.add_argument("--enrichment", default="Log2_Enrichment", metavar="COLUMN",
                    help=(
                        "Column containing log2-fold panning enrichment scores. "
                        "Carried unchanged to the output CSV so results can be ranked "
                        "by both structural confidence and biological enrichment. "
                        "Pass an empty string to skip. Default: Log2_Enrichment"
                    ))

    # ---- Antigen ----
    ag = parser.add_argument_group("Antigen")
    ag.add_argument("--antigen", required=True, metavar="PATH",
                    help="Antigen structure file (.pdb or .cif). Required.")
    ag.add_argument("--antigen-chain", default=None, metavar="ID",
                    help=(
                        "Single chain ID to use from the antigen file. "
                        "Default: first chain found."
                    ))
    ag.add_argument("--antigen-chains", nargs="+", default=None, metavar="ID",
                    help=(
                        "Multiple chain IDs to merge into a single antigen chain, "
                        "e.g. --antigen-chains A B. Residues are renumbered sequentially; "
                        "coordinates are preserved. Overrides --antigen-chain."
                    ))
    ag.add_argument("--use-template", action="store_true",
                    help=(
                        "Provide the antigen structure as a Boltz2 structural template "
                        "for chain B during docking (Stage 3). Strongly recommended when "
                        "the antigen structure is experimentally determined (PDB/AlphaFold). "
                        "Omit only if the antigen is itself an uncertain prediction."
                    ))

    # ---- Stage 2: docking ----
    dock = parser.add_argument_group("Stage 2 — Rigid-body Boltz2 docking")
    dock.add_argument("--num-models", type=int, default=5, metavar="N",
                      help=(
                          "Number of Boltz2 diffusion samples for the rigid-body docking run. "
                          "All samples use the rank-0 OpenMM-relaxed VHH as a fixed template. "
                          "Mean ± std are reported across samples. Default: 5"
                      ))
    dock.add_argument("--max-parallel-samples", type=int, default=None, metavar="N",
                      help=(
                          "Maximum number of docking diffusion samples to run in parallel "
                          "on the GPU. Reduce if the GPU runs out of memory. "
                          "Default: equal to --num-models."
                      ))
    dock.add_argument("--recycling-steps", type=int, default=3, metavar="N",
                      help=(
                          "Number of Boltz2 recycling iterations per diffusion sample. "
                          "Higher values can improve structure/confidence convergence at "
                          "added compute cost per sample. Default: 3"
                      ))

    # ---- Hardware / MSA ----
    hw = parser.add_argument_group("Hardware and MSA")
    hw.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "tpu"],
                    help="Hardware accelerator for Boltz2. Default: gpu")
    hw.add_argument("--no-msa-server", action="store_true",
                    help=(
                        "Disable the ColabFold MSA server for the antigen (chain B). "
                        "The VHH (chain A) never uses the MSA server regardless of this "
                        "flag — it always runs in single-sequence mode since its "
                        "conformation is already fixed by the NanobodyBuilder2 template. "
                        "Use only when pre-computed MSAs are available or for offline runs."
                    ))

    args = parser.parse_args()

    # ---- Antigen setup ----
    antigen_path = Path(args.antigen)
    if not antigen_path.exists():
        sys.exit(f"Error: antigen file not found — {args.antigen}")

    antigen_model = load_structure(antigen_path)
    antigen_seqs_all = extract_all_sequences(antigen_model)
    if not antigen_seqs_all:
        sys.exit("Error: no protein chains found in antigen structure.")

    requested_chains = args.antigen_chains or (
        [args.antigen_chain] if args.antigen_chain else None
    )
    if requested_chains:
        missing = [c for c in requested_chains if c not in antigen_seqs_all]
        if missing:
            sys.exit(
                f"Error: chain(s) {missing} not found. "
                f"Available: {list(antigen_seqs_all.keys())}"
            )
        if len(requested_chains) == 1:
            antigen_chain_id = requested_chains[0]
            antigen_seq = antigen_seqs_all[antigen_chain_id]
            antigen_cif = prepare_antigen_cif(antigen_path)
        else:
            antigen_chain_id = "+".join(requested_chains)
            antigen_cif, antigen_seq = merge_antigen_chains(antigen_path, requested_chains)
    else:
        antigen_chain_id, antigen_seq = next(iter(antigen_seqs_all.items()))
        antigen_cif = prepare_antigen_cif(antigen_path)

    print(f"Antigen: {antigen_path.name}  chain={antigen_chain_id}  ({len(antigen_seq)} residues)")
    print(f"  {antigen_seq[:80]}{'...' if len(antigen_seq) > 80 else ''}\n")

    # ---- Load input ----
    df = load_input(args.input, args.names, args.sequences)
    total = len(df)

    enrichment_col = args.enrichment if args.enrichment else None
    if enrichment_col and enrichment_col not in df.columns:
        print(f"Warning: enrichment column '{enrichment_col}' not found — omitted from output.")
        enrichment_col = None
    summary_cols = (
        SUMMARY_COLUMNS if enrichment_col
        else [c for c in SUMMARY_COLUMNS if c != "Log2_Enrichment"]
    )

    print(f"Loaded {total} VHH cluster representatives from '{args.input}'")
    print(f"  num-models={args.num_models}  recycling-steps={args.recycling_steps}  accelerator={args.accelerator}")
    print(f"  use-template={'yes' if args.use_template else 'no'}\n")

    # ---- Output directories ----
    out_dir = Path(args.output) if args.output else default_output_dir(args.input)
    vhh_dir  = out_dir / "vhh_structures"
    dock_dir = out_dir / "docking"
    log_dir  = out_dir / "logs"
    for d in (out_dir, vhh_dir, dock_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    summary_rows:   list[dict] = []
    per_model_rows: list[dict] = []

    vhh_bar = tqdm(
        df.iterrows(), total=total,
        desc="Processing VHHs", unit="VHH", ncols=110,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}",
    )

    for i, row in vhh_bar:
        seq  = clean_sequence(row[args.sequences])
        name = make_safe_name(row[args.names])
        enrichment = (
            float(row[enrichment_col])
            if enrichment_col and pd.notna(row.get(enrichment_col))
            else None
        )

        if not seq:
            tqdm.write(f"  Skipping '{name}' — empty sequence.")
            continue

        vhh_bar.set_description(f"VHH: {name}")
        tqdm.write(f"\n{'='*70}")
        tqdm.write(
            f"[{i+1}/{total}] {name}  ({len(seq)} aa)"
            + (f"  log2_enrichment={enrichment:.2f}" if enrichment is not None else "")
        )
        tqdm.write(f"{'='*70}")

        name_vhh_dir = vhh_dir / name
        name_vhh_dir.mkdir(exist_ok=True)

        # Stage 1 — NanobodyBuilder2: best model only, OpenMM-relaxed
        tqdm.write("\n  -- Stage 1: NanobodyBuilder2 best model --")
        vhh_bar.set_postfix_str("Stage 1: NanobodyBuilder2")
        best_model_cif = run_immunebuilder_best(seq, name, name_vhh_dir)

        if best_model_cif is None:
            tqdm.write(f"  No VHH structure generated for '{name}' — skipping.")
            continue

        # Stage 2 — Rigid-body Boltz2 docking (num_models diffusion samples)
        tqdm.write(
            f"\n  -- Stage 2: Rigid-body Boltz2 docking "
            f"({args.num_models} diffusion samples, recycling_steps={args.recycling_steps}) --"
        )
        vhh_bar.set_postfix_str("Stage 2: Boltz2 docking")
        all_dock_rows = dock_vhh(
            name=name,
            best_model_cif=best_model_cif,
            binder_seq=seq,
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

        per_model_rows.extend(all_dock_rows)
        pd.DataFrame(per_model_rows).reindex(columns=PER_MODEL_COLUMNS).to_csv(
            out_dir / "ensemble_per_model.csv", index=False
        )

        # Stage 3 — Convergence scoring across diffusion samples
        tqdm.write("\n  -- Stage 3: Convergence scoring --")
        vhh_bar.set_postfix_str("Stage 3: convergence")

        valid_rows = [r for r in all_dock_rows if r.get("binding_score") is not None]
        if not valid_rows:
            tqdm.write(f"  No valid docking results for '{name}'.")
            continue

        scores = [r["binding_score"] for r in valid_rows if r["binding_score"] is not None]
        iptms  = [r["iptm"]          for r in valid_rows if r.get("iptm")          is not None]
        paes   = [r["pae_interface"] for r in valid_rows if r.get("pae_interface") is not None]
        bsas   = [r["bsa_A2"]        for r in valid_rows if r.get("bsa_A2")        is not None]

        epitope_overlap = compute_epitope_overlap(valid_rows)
        pose_rmsd = compute_pose_convergence_rmsd(dock_dir, name, args.num_models)

        mean_bs  = round(float(np.mean(scores)), 4) if scores else None
        best_bs  = round(float(max(scores)), 4)     if scores else None
        std_bs   = round(float(np.std(scores)), 4)  if len(scores) > 1 else None
        mean_ip  = round(float(np.mean(iptms)), 4)  if iptms else None
        best_ip  = round(float(max(iptms)), 4)      if iptms else None
        mean_pa  = round(float(np.mean(paes)), 4)   if paes else None
        mean_bsa = round(float(np.mean(bsas)), 2)   if bsas else None

        conv_rank = (
            round(epitope_overlap * mean_bs, 4)
            if epitope_overlap is not None and mean_bs is not None
            else None
        )

        summary_rows.append({
            "Cluster":                  name,
            "sequence":                 seq,
            "Log2_Enrichment":          enrichment,
            "n_conformers_kept":        1,
            "n_docking_runs":           len(all_dock_rows),
            "mean_binding_score":       mean_bs,
            "best_binding_score":       best_bs,
            "binding_score_std":        std_bs,
            "mean_iptm":                mean_ip,
            "best_iptm":                best_ip,
            "mean_pae_interface":       mean_pa,
            "mean_bsa_A2":              mean_bsa,
            "epitope_overlap_fraction": epitope_overlap,
            "pose_convergence_rmsd":    pose_rmsd,
            "convergence_rank":         conv_rank,
        })

        pd.DataFrame(summary_rows).reindex(columns=summary_cols).to_csv(
            out_dir / "ensemble_binding_scores.csv", index=False
        )

        tqdm.write(
            f"\n  Result:  binding_score={mean_bs}  iptm={mean_ip}  "
            f"epitope_overlap={epitope_overlap}  convergence_rank={conv_rank}"
        )
        vhh_bar.set_postfix_str(f"score={mean_bs}  overlap={epitope_overlap}  rank={conv_rank}")

    vhh_bar.close()

    if not summary_rows:
        print("\nNo results generated.", file=sys.stderr)
        sys.exit(1)

    results_df = (
        pd.DataFrame(summary_rows)
        .reindex(columns=summary_cols)
        .sort_values("convergence_rank", ascending=False, na_position="last")
    )
    results_df.to_csv(out_dir / "ensemble_binding_scores.csv", index=False)

    pm_df = pd.DataFrame(per_model_rows).reindex(columns=PER_MODEL_COLUMNS)
    print_final_report(results_df, df, pm_df, out_dir, args.names)


if __name__ == "__main__":
    main()
