#!/usr/bin/env python3
"""
Protein biophysical analysis script.

Reads protein sequences from a CSV or Excel file and computes:
  - Molecular weight
  - Isoelectric point (pI)
  - GRAVY score
  - Aromaticity
  - Aliphatic index
  - Extinction coefficient (reduced and oxidized cysteines)

Usage:
    python biophysical_analysis.py input.csv
    python biophysical_analysis.py input.xlsx --column ProteinSeq
    python biophysical_analysis.py input.csv --output results.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# ---------------------------------------------------------------------------
# Calculation blocks
# ---------------------------------------------------------------------------

def calc_molecular_weight(pa: ProteinAnalysis) -> float:
    """Molecular weight in Daltons."""
    return pa.molecular_weight()


def calc_isoelectric_point(pa: ProteinAnalysis) -> float:
    """Theoretical isoelectric point (pI)."""
    return pa.isoelectric_point()


def calc_gravy(pa: ProteinAnalysis) -> float:
    """Grand Average of Hydropathicity (GRAVY) score."""
    return pa.gravy()


def calc_aromaticity(pa: ProteinAnalysis) -> float:
    """Fraction of aromatic residues (Phe, Trp, Tyr)."""
    return pa.aromaticity()


def calc_aliphatic_index(pa: ProteinAnalysis) -> float:
    """
    Aliphatic index (Ikai 1980).
    AI = X(Ala) + 2.9*X(Val) + 3.9*(X(Ile) + X(Leu))
    where X is mole percent of each residue.
    """
    counts = pa.count_amino_acids()
    length = len(pa.sequence)
    ala = counts.get("A", 0) / length * 100
    val = counts.get("V", 0) / length * 100
    ile = counts.get("I", 0) / length * 100
    leu = counts.get("L", 0) / length * 100
    return ala + 2.9 * val + 3.9 * (ile + leu)


def calc_extinction_coefficient(pa: ProteinAnalysis) -> tuple[float, float]:
    """
    Molar extinction coefficient at 280 nm (M⁻¹ cm⁻¹).
    Returns (reduced_cysteines, oxidized_cysteines).
    """
    return pa.molar_extinction_coefficient()


# ---------------------------------------------------------------------------
# Sequence analysis orchestrator
# ---------------------------------------------------------------------------

def analyze_sequence(sequence: str) -> dict:
    pa = ProteinAnalysis(sequence)
    ext_reduced, ext_oxidized = calc_extinction_coefficient(pa)
    return {
        "Molecular Weight (Da)":              round(calc_molecular_weight(pa), 2),
        "pI":                                 round(calc_isoelectric_point(pa), 2),
        "GRAVY":                              round(calc_gravy(pa), 4),
        "Aromaticity":                        round(calc_aromaticity(pa), 4),
        "Aliphatic Index":                    round(calc_aliphatic_index(pa), 2),
        "Ext. Coeff Reduced (M-1 cm-1)":      ext_reduced,
        "Ext. Coeff Oxidized (M-1 cm-1)":     ext_oxidized,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

RESULT_COLUMNS = [
    "Molecular Weight (Da)",
    "pI",
    "GRAVY",
    "Aromaticity",
    "Aliphatic Index",
    "Ext. Coeff Reduced (M-1 cm-1)",
    "Ext. Coeff Oxidized (M-1 cm-1)",
]

EMPTY_RESULT = {col: None for col in RESULT_COLUMNS}


def load_input(filepath: str, column: str) -> pd.DataFrame:
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

    if column not in df.columns:
        sys.exit(
            f"Error: column '{column}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    return df


def default_output_path(input_path: str) -> str:
    p = Path(input_path)
    return str(p.parent / f"{p.stem}_biophysical_results.csv")


def clean_sequence(raw: object) -> str:
    return str(raw).strip().upper().replace(" ", "").replace("\n", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute biophysical properties for protein sequences."
    )
    parser.add_argument("input", help="Input CSV or Excel file")
    parser.add_argument(
        "--column",
        default="Sequence",
        help="Column containing protein sequences (default: Sequence)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output CSV path. Defaults to the same directory as the input file, "
            "named <input>_biophysical_results.csv"
        ),
    )
    args = parser.parse_args()

    df = load_input(args.input, args.column)
    total = len(df)
    print(f"Loaded {total} sequences from '{args.input}' (column: '{args.column}')")

    results = []
    errors = 0
    for i, raw_seq in enumerate(df[args.column], start=1):
        seq = clean_sequence(raw_seq)
        try:
            results.append(analyze_sequence(seq))
        except Exception as exc:
            print(f"  [row {i}] Warning: could not analyze '{seq[:30]}...' — {exc}", file=sys.stderr)
            results.append(EMPTY_RESULT.copy())
            errors += 1

    output_df = pd.concat([df, pd.DataFrame(results)], axis=1)
    output_path = args.output or default_output_path(args.input)
    output_df.to_csv(output_path, index=False)

    print(f"Done. {total - errors}/{total} sequences analyzed successfully.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
