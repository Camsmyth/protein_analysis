#!/usr/bin/env python3
"""
Simple Boltz2 rigid-body docking of a nanobody against an antigen.

Takes a nanobody structure and an antigen structure, converts PDBs to CIF as
needed, and runs a Boltz2 templated complex prediction. Both structures are
provided as templates (chain A = nanobody, chain B = antigen).

Multiple antigen chains can be merged into a single chain via --antigen-chains,
e.g. for Fc homodimers where both heavy chains form the binding interface.

Usage:
    python boltz_dock.py --nanobody my_vhh.pdb --antigen antigen.pdb
    python boltz_dock.py --nanobody my_vhh.pdb --antigen Ab.pdb \\
        --antigen-chains B D --output docking_results/
    python boltz_dock.py --nanobody my_vhh.pdb --antigen antigen.pdb \\
        --models 3 --use-msa-server
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gemmi
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1

BOLTZ_BIN = Path(sys.executable).parent / "boltz"


# ---------------------------------------------------------------------------
# Structure utilities
# ---------------------------------------------------------------------------

def load_structure(path: Path):
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    elif suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        sys.exit(f"Error: unsupported format '{suffix}'. Use .pdb or .cif.")
    structure = parser.get_structure(path.stem, str(path))
    return next(iter(structure))


def extract_sequence(path: Path, chain_id: str | None = None) -> tuple[str, str]:
    """Return (chain_id, sequence) for the requested chain, or the first chain."""
    model = load_structure(path)
    for chain in model.get_chains():
        if chain_id is None or chain.id == chain_id:
            seq = "".join(
                seq1(r.resname) for r in chain.get_residues() if r.id[0] == " "
            )
            if seq:
                return chain.id, seq
    sys.exit(
        f"Error: {'chain ' + chain_id if chain_id else 'no protein chain'} "
        f"found in {path.name}"
    )


def pdb_to_cif(pdb_path: Path) -> Path:
    """
    Convert a PDB to CIF, filling entity_poly_seq from ATOM records so Boltz2
    can parse it as a template. Saved alongside the original and reused on
    subsequent runs.
    """
    if pdb_path.suffix.lower() in (".cif", ".mmcif"):
        return pdb_path

    cif_path = pdb_path.with_suffix(".cif")
    if cif_path.exists():
        print(f"  Using cached CIF: {cif_path.name}")
        return cif_path

    st = gemmi.read_structure(str(pdb_path))
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
    print(f"  Converted {pdb_path.name} → {cif_path.name}")
    return cif_path


def merge_chains(antigen_path: Path, chain_ids: list[str]) -> tuple[Path, str]:
    """
    Merge multiple antigen chains into a single chain A PDB, renumbering
    residues sequentially. Coordinates are preserved exactly.
    Returns (merged_pdb_path, merged_sequence).
    """
    label = "".join(chain_ids)
    out_pdb = antigen_path.parent / f"{antigen_path.stem}_merged{label}.pdb"

    model = load_structure(antigen_path)
    available = [c.id for c in model.get_chains()]
    missing = [c for c in chain_ids if c not in available]
    if missing:
        sys.exit(
            f"Error: chain(s) {missing} not found in {antigen_path.name}. "
            f"Available: {available}"
        )

    all_residues = []
    for cid in chain_ids:
        all_residues.extend(r for r in model[cid] if r.id[0] == " ")

    if not all_residues:
        sys.exit(f"Error: no ATOM residues in chains {chain_ids} of {antigen_path.name}")

    lines = []
    serial = 1
    for res_num, res in enumerate(all_residues, start=1):
        for atom in res.get_atoms():
            x, y, z = atom.coord
            lines.append(
                f"ATOM  {serial:5d} {atom.name:<4s} {res.resname:<3s} A{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{atom.occupancy:6.2f}{atom.bfactor:6.2f}          "
                f"{atom.element:>2s}\n"
            )
            serial += 1
    lines.append("END\n")
    out_pdb.write_text("".join(lines))

    merged_seq = "".join(seq1(r.resname) for r in all_residues)
    print(f"  Merged chains {chain_ids} → {out_pdb.name} ({len(all_residues)} residues)")
    return out_pdb, merged_seq


def write_complex_yaml(
    nanobody_seq: str,
    antigen_seq: str,
    path: Path,
    nanobody_cif: Path,
    antigen_cif: Path,
    use_template: bool,
) -> None:
    """Write Boltz2 two-chain YAML. Chain A = nanobody, chain B = antigen."""
    lines = [
        "sequences:",
        "  - protein:",
        "      id: A",
        f'      sequence: "{nanobody_seq}"',
        "  - protein:",
        "      id: B",
        f'      sequence: "{antigen_seq}"',
    ]
    if use_template:
        nb_key = "cif" if nanobody_cif.suffix.lower() in (".cif", ".mmcif") else "pdb"
        ag_key = "cif" if antigen_cif.suffix.lower()  in (".cif", ".mmcif") else "pdb"
        lines += [
            "templates:",
            f"  - {nb_key}: {nanobody_cif.resolve()}",
            "    chain_id: [A]",
            f"  - {ag_key}: {antigen_cif.resolve()}",
            "    chain_id: [B]",
        ]
    path.write_text("\n".join(lines) + "\n")


def run_boltz(
    yaml_dir: Path,
    out_dir: Path,
    accelerator: str,
    num_models: int,
    use_msa_server: bool,
    max_parallel: int,
) -> int:
    cmd = [
        str(BOLTZ_BIN), "predict", str(yaml_dir),
        "--out_dir", str(out_dir),
        "--accelerator", accelerator,
        "--diffusion_samples", str(num_models),
        "--model", "boltz2",
        "--write_full_pae",
        "--num_workers", "4",
        "--preprocessing-threads", "4",
        "--max_parallel_samples", str(max_parallel),
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def parse_confidence(pred_dir: Path, run_name: str, num_models: int) -> None:
    print("\nConfidence scores:")
    for i in range(num_models):
        conf = pred_dir / f"confidence_{run_name}_model_{i}.json"
        if not conf.exists():
            print(f"  model {i}: confidence file not found")
            continue
        data = json.loads(conf.read_text())
        iptm = data.get("iptm")
        ptm  = data.get("ptm")
        if isinstance(iptm, float) and isinstance(ptm, float):
            bs = 0.8 * iptm + 0.2 * ptm
            print(f"  model {i}: binding_score={bs:.3f}  ipTM={iptm:.3f}  pTM={ptm:.3f}")
        else:
            print(f"  model {i}: scores unavailable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Boltz2 nanobody–antigen docking with structural templates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nanobody", required=True, metavar="PATH",
                        help="Nanobody/VHH structure (.pdb or .cif). Required.")
    parser.add_argument("--antigen", required=True, metavar="PATH",
                        help="Antigen structure (.pdb or .cif). Required.")
    parser.add_argument("--nanobody-chain", default=None, metavar="ID",
                        help="Chain ID to use from nanobody file. Default: first chain.")

    ag = parser.add_mutually_exclusive_group()
    ag.add_argument("--antigen-chain", default=None, metavar="ID",
                    help="Single chain ID to use from antigen file. Default: first chain.")
    ag.add_argument("--antigen-chains", nargs="+", default=None, metavar="ID",
                    help="Multiple chain IDs to merge into a single antigen chain, "
                         "e.g. --antigen-chains B D for an Fc homodimer. "
                         "Residues are concatenated and renumbered sequentially.")

    parser.add_argument("--output", default=None, metavar="DIR",
                        help="Output directory. Default: boltz_docking/ next to antigen file.")
    parser.add_argument("--models", type=int, default=3, metavar="N",
                        help="Number of Boltz2 diffusion samples. Default: 3")
    parser.add_argument("--max-parallel", type=int, default=1, metavar="N",
                        help="Max parallel GPU samples. Default: 1 (sequential, avoids OOM). "
                             "Increase if your GPU has enough VRAM.")
    parser.add_argument("--no-template", action="store_true",
                        help="Do not provide structures as Boltz2 templates "
                             "(free docking, no structural constraints).")
    parser.add_argument("--no-msa-server", action="store_true",
                        help="Disable ColabFold MSA server (enabled by default).")
    parser.add_argument("--accelerator", default="gpu",
                        choices=["gpu", "cpu", "tpu"],
                        help="Hardware accelerator. Default: gpu")
    args = parser.parse_args()

    nanobody_path = Path(args.nanobody)
    antigen_path  = Path(args.antigen)
    for p in (nanobody_path, antigen_path):
        if not p.exists():
            sys.exit(f"Error: file not found — {p}")

    out_dir = Path(args.output) if args.output else antigen_path.parent / "boltz_docking"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing structures…")

    # Nanobody
    nanobody_cif = pdb_to_cif(nanobody_path)
    nb_chain, nb_seq = extract_sequence(nanobody_path, args.nanobody_chain)
    print(f"  Nanobody chain {nb_chain}: {len(nb_seq)} residues")

    # Antigen — single chain, merged chains, or first chain
    if args.antigen_chains:
        merged_pdb, ag_seq = merge_chains(antigen_path, args.antigen_chains)
        antigen_cif = pdb_to_cif(merged_pdb)
        ag_label = "+".join(args.antigen_chains)
    elif args.antigen_chain:
        antigen_cif = pdb_to_cif(antigen_path)
        ag_label, ag_seq = extract_sequence(antigen_path, args.antigen_chain)
    else:
        antigen_cif = pdb_to_cif(antigen_path)
        ag_label, ag_seq = extract_sequence(antigen_path, None)

    print(f"  Antigen chain(s) {ag_label}: {len(ag_seq)} residues")
    print(f"  Total complex: {len(nb_seq) + len(ag_seq)} residues")

    run_name = f"{nanobody_path.stem}_vs_{antigen_path.stem}"
    boltz_out = out_dir / "boltz_raw"
    boltz_out.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as yaml_tmp:
        yaml_path = Path(yaml_tmp) / f"{run_name}.yaml"
        write_complex_yaml(
            nb_seq, ag_seq, yaml_path,
            nanobody_cif, antigen_cif,
            use_template=not args.no_template,
        )
        rc = run_boltz(
            Path(yaml_tmp), boltz_out, args.accelerator,
            args.models, not args.no_msa_server, args.max_parallel,
        )

    if rc != 0:
        sys.exit(f"Error: Boltz2 failed (exit code {rc}). Check output above.")

    pred_dirs = list(boltz_out.glob(f"*/predictions/{run_name}"))
    if not pred_dirs:
        sys.exit("Error: Boltz2 ran but no predictions directory found.")

    pred_dir = pred_dirs[0]
    parse_confidence(pred_dir, run_name, args.models)

    cifs = sorted(pred_dir.glob(f"{run_name}_model_*.cif"))
    for cif in cifs:
        shutil.copy2(cif, out_dir / cif.name)

    print(f"\nDone. {len(cifs)} complex structure(s) written to: {out_dir}/")
    for cif in cifs:
        print(f"  {cif.name}")


if __name__ == "__main__":
    main()
