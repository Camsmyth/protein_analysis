================================================================================
 PROTEIN ANALYSIS PIPELINE
================================================================================

REQUIREMENTS
------------
- Python 3.10 or newer
- Internet access (for MSA server during Boltz predictions, unless noted)
- GPU strongly recommended (CUDA on Linux/Windows)
- First run downloads ~3 GB of Boltz2 model weights to ~/.boltz (cached)
- First run downloads ImmuneBuilder model weights (~500 MB, cached)


SETUP (run once on any new workstation)
----------------------------------------
  bash setup.sh

This creates a virtual environment called 'protein/' and installs all
dependencies including ImmuneBuilder (NanobodyBuilder2), Boltz2, and ANARCI.
Do NOT copy the 'protein/' folder between machines — always run setup.sh
on the target machine instead.

Activate the environment before running any script:
  source protein/bin/activate          # macOS / Linux
  protein\Scripts\activate             # Windows


SCRIPTS OVERVIEW
-----------------
  ensemble_pipeline.py      Main pipeline: VHH-antigen docking + convergence
                             scoring for enriched phage display candidates.
  interface_remodeling.py   CDR alanine/glycine scan + hotspot-guided
                             mutation optimisation, starting from a validated
                             ensemble_pipeline.py complex.
  manual_mutant_scan.py     Score your own hand-designed mutant sequences
                             (FASTA) against a validated WT complex, using
                             the same WT-pose-templated docking strategy as
                             interface_remodeling.py.
  biophysical_analysis.py   Sequence-level biophysical properties (MW, pI,
                             GRAVY, etc.) — no structure prediction.
  nanobody_structure.py     Standalone NanobodyBuilder2 runner (structures
                             only, no docking) for a FASTA of VHH sequences.
  boltz_dock.py              Lightweight one-off Boltz2 docking of a single
                             nanobody structure file against an antigen —
                             use when you already have a VHH structure and
                             don't need the full NanobodyBuilder2 pipeline.
  boltz_pipeline.py          Bare Boltz2 single-chain structure prediction
                             from a CSV/Excel of sequences — no docking, no
                             NanobodyBuilder2. Optional CPU/GPU usage
                             monitoring via monitor.py.
  monitor.py                  Shared CPU/GPU usage monitor (macOS: asitop +
                             powermetrics; Linux: /proc/stat + nvidia-smi).
                             Imported by boltz_pipeline.py; not run directly.

Most users only need ensemble_pipeline.py (triage) and, for promising hits,
interface_remodeling.py (automated mutation optimisation) or
manual_mutant_scan.py (scoring your own hand-picked mutants). The others are
standalone utilities kept for lighter-weight or exploratory use.


================================================================================
1. ensemble_pipeline.py — MAIN PIPELINE
================================================================================

VHH–antigen rigid-body docking pipeline for enriched phage display cluster
representatives. Takes a CSV/Excel of VHH sequences (with enrichment scores)
and an antigen structure, and produces convergence-scored complex predictions.

Background:
  VHH sequences entering this pipeline are cluster representatives selected
  by Levenshtein distance (0.85 threshold) from ONT long-read sequencing of
  a phage display experiment, filtered for log2-fold enrichment between
  panning rounds. The pipeline is the final computational triage step before
  wet-lab ordering.

  NanobodyBuilder2 generates 4 structures from 4 independently pre-trained
  networks. The rank-0 (highest-confidence / closest-to-consensus) structure
  is OpenMM-relaxed and used as the VHH template for all Boltz2 docking.
  Templating both chains (VHH + antigen) implements rigid-body docking: the
  VHH conformation is fixed, and the diffusion process samples interface
  geometry. Convergence across diffusion samples is the primary confidence
  signal.

  Chain A (the VHH) never queries the MSA server, regardless of
  --no-msa-server — it always runs in single-sequence mode, since its
  conformation is already fixed by the NanobodyBuilder2 template and
  nanobody CDR loops carry little useful coevolutionary signal. This also
  keeps novel/proprietary VHH sequences from being sent to the remote MSA
  server. --no-msa-server only affects chain B (the antigen).

Three automated stages:
  Stage 1  NanobodyBuilder2 (ImmuneBuilder) — generates 4 model-diverse VHH
           structures; selects and OpenMM-relaxes the rank-0 (best) model.
  Stage 2  Rigid-body Boltz2 docking — runs a full two-chain (VHH + antigen)
           complex prediction using the rank-0 VHH as a fixed structural
           template for chain A. --num-models diffusion samples are
           generated in a single Boltz2 call (default: 5), with
           --recycling-steps recycling iterations (default: 3).
  Stage 3  Convergence scoring — computes epitope overlap (fraction of antigen
           residues contacted by ≥50% of diffusion samples) and pose RMSD
           across samples. convergence_rank = epitope_overlap ×
           mean_binding_score is the primary output metric.

Experimental — confidence-guided flexible template (Stage 1, opt-in):
  --flexible-cdr-template replaces the fully-rigid VHH template with one
  that strips residues where NanobodyBuilder2's own 4-model ensemble
  disagrees (per-residue Cα spread above --flexible-template-cutoff,
  default 2.0 Å), leaving those residues free for Boltz2 to diffuse instead
  of frozen at NanobodyBuilder2's possibly-wrong guess (this typically, but
  not always, affects CDR3, the least-constrained loop). Off by default —
  compare against the default rigid-template behaviour before relying on
  this for triage decisions; the default rigid-template code path is
  completely unaffected when this flag is not passed.

Input CSV/Excel (one row per cluster representative):
  Cluster_R2           — cluster name / ID      (--names)
  Protein_Sequence_R2  — VHH amino acid sequence (--sequences)
  Log2_Enrichment      — panning enrichment score (--enrichment, optional)

Typical usage:
  python ensemble_pipeline.py h7/h7_R1vsR2_known_enriched.csv \
    --antigen h7/hCD7_alphafold.pdb \
    --names Cluster_R2 \
    --sequences Protein_Sequence_R2 \
    --use-template

With the experimental flexible template:
  python ensemble_pipeline.py files/CD7_clones_IDT.xlsx \
    --names Name --sequences Sequence \
    --antigen binding/h7/hCD7_alphafold.pdb \
    --use-template \
    --flexible-cdr-template

All flags and defaults:

INPUT / OUTPUT
  input (positional)
    CSV or Excel file containing VHH cluster representatives. Required.

  --output DIR
    Root output directory. All stages write here.
    Default: ensemble_predictions/ in the same directory as the input file.

  --names COLUMN
    Column containing the VHH name / cluster ID.
    Default: Cluster

  --sequences COLUMN
    Column containing VHH amino acid sequences.
    Default: Protein_Sequence_R2

  --enrichment COLUMN
    Column containing log2-fold panning enrichment scores. Carried unchanged
    to the output CSV so results can be ranked by both structural confidence
    and biological enrichment. If the column is absent from the input file
    it is silently omitted from the output (no error).
    Default: Log2_Enrichment

ANTIGEN
  --antigen PATH
    Antigen structure file (.pdb or .cif). Required.
    PDB files are automatically converted to CIF for Boltz2; the converted
    file is saved alongside the original and reused on subsequent runs.

  --antigen-chain ID
    Single chain ID to use from the antigen file.
    Default: first chain found in the file.

  --antigen-chains ID [ID ...]
    Multiple chain IDs to merge into a single antigen chain, e.g.:
      --antigen-chains A B
    Residues from each chain are concatenated in order and renumbered
    sequentially 1, 2, 3 ... while preserving original coordinates.
    The merged structure is saved alongside the original and reused on
    subsequent runs. Overrides --antigen-chain.

  --use-template
    Provide the antigen structure as a Boltz2 structural template for chain B
    during Stage 2 docking. Strongly recommended when the antigen structure
    is experimentally determined (crystal structure, cryo-EM, or high-
    confidence AlphaFold model). Omit only if the antigen is itself an
    uncertain computational prediction that you do not want to constrain.
    Default: off (flag absent = no antigen template)

STAGE 1 — CONFIDENCE-GUIDED FLEXIBLE TEMPLATE (experimental, opt-in)
  --flexible-cdr-template
    Strip high-uncertainty residues (per NanobodyBuilder2's own 4-model Cα
    spread) from the VHH template before Boltz2 docking, instead of rigidly
    templating the entire VHH. Off by default; the standard rigid-template
    code path is unaffected when this flag is omitted.

  --flexible-template-cutoff ANGSTROM
    Per-residue 4-model Cα spread (Å) above which a residue is stripped from
    the template. Only used with --flexible-cdr-template.
    Default: 2.0

  --flexible-template-min-run N
    Minimum consecutive high-uncertainty residues required before stripping
    them (suppresses stripping isolated single-residue noise).
    Default: 1

STAGE 2 — RIGID-BODY BOLTZ2 DOCKING
  --num-models N
    Number of Boltz2 diffusion samples for the rigid-body docking run. All
    samples use the rank-0 OpenMM-relaxed VHH as a fixed structural template.
    Mean ± std are reported across samples.
    Default: 5

  --recycling-steps N
    Number of Boltz2 recycling iterations per diffusion sample. Higher
    values can improve structure/confidence convergence at added compute
    cost per sample.
    Default: 3

  --max-parallel-samples N
    Maximum number of diffusion samples to run simultaneously on the GPU.
    Reduce if CUDA runs out of memory (OOM errors).
    Default: equal to --num-models (all samples in parallel).

HARDWARE AND MSA
  --accelerator gpu|cpu|tpu
    Hardware accelerator passed to Boltz2.
    Default: gpu

  --no-msa-server
    Disable the ColabFold MSA server for the antigen (chain B) only — the
    VHH (chain A) never uses the MSA server regardless of this flag, since
    its conformation is already fixed by the NanobodyBuilder2 template.
    MSA fetching may be rate-limited on the public server; this is normal.
    Default: off (MSA server enabled for the antigen)

Outputs (under --output/):
  vhh_structures/{name}/
    {name}_best_model.pdb          — rank-0 NanobodyBuilder2 structure,
                                     OpenMM-relaxed (used as docking template)
    {name}_best_model.cif          — CIF conversion for Boltz2
    {name}_flexible_template.cif   — only when --flexible-cdr-template
                                     stripped one or more residues

  docking/{name}/
    {name}_model_{m}.cif           — Boltz2 complex structures (1 per sample)
    Boltz2 confidence JSONs and PAE .npz files

  logs/
    {name}.log                     — Boltz2 stdout/stderr for the docking run

  ensemble_binding_scores.csv
    One row per VHH, sorted by convergence_rank (highest first).
    Columns:
      Cluster                — representative name
      sequence               — VHH amino acid sequence
      Log2_Enrichment        — carried from input; biological enrichment signal
                               (omitted if column not present in input CSV)
      n_conformers_kept      — always 1 (rank-0 best model)
      n_docking_runs         — number of Boltz2 diffusion samples scored
      mean_binding_score     — 0.8×ipTM + 0.2×pTM averaged across all samples
                               (>0.6 reasonable, >0.7 strong)
      best_binding_score     — highest single sample score
      binding_score_std      — standard deviation across samples
      mean_iptm              — mean interface TM-score (>0.7 = strong binding)
      best_iptm              — highest single sample ipTM
      mean_pae_interface     — mean cross-chain PAE at interface (lower = better)
      mean_bsa_A2            — mean buried surface area (Å²)
                               600–1000 Å² = good VHH interface
                               >1000 Å² = verify (may indicate non-specific burial)
                               <600 Å² = weak contact
      epitope_overlap_fraction — fraction of antigen interface residues shared
                                 across ≥50% of diffusion samples. High value
                                 means samples consistently converge on the
                                 same antigen epitope.
      pose_convergence_rmsd  — Cα RMSD of the VHH in complex after superimposing
                                on antigen (low = consistent binding pose)
      convergence_rank       — epitope_overlap_fraction × mean_binding_score;
                                primary ranking metric; combines binding quality
                                with pose consistency
      n_flexible_residues_stripped — count of residues removed from the VHH
                                template (only populated when
                                --flexible-cdr-template was used, else blank)
      flexible_residues_stripped   — comma-separated (resnum+icode) list of
                                those residues (same conditional population)

  ensemble_per_model.csv
    Full per-Boltz2-sample detail with all confidence metrics, interface
    contacts, BSA, and clash count for every diffusion sample.

  Final report (printed to terminal):
    Human-readable summary table with per-VHH tier assignment:
      READY    — high binding score, ipTM, epitope overlap, pose convergence,
                 no clashes; proceed to wet-lab ordering
      REVIEW   — intermediate metrics; inspect structures before ordering
      UNCERTAIN — low confidence or high clashes; further validation needed
    BSA is shown as a contextual flag (not used as a gate):
      ✓ good (600–1000 Å²)
      ⚠ verify interface (>1000 Å²)   [may be non-specific hydrophobic burial]
      ✗ weak (<600 Å²)
    When --flexible-cdr-template stripped residues for a VHH, a line noting
    how many and which ones is shown alongside the other diagnostics.

Interpretation:
  Strong ordering candidate:
    High convergence_rank (e.g. >0.5) + high Log2_Enrichment + tier READY
  Uncertain / do not order without further validation:
    Low epitope_overlap_fraction (<0.3) even with high mean_binding_score
    High pose_convergence_rmsd (>5 Å) — poses diverge across diffusion samples
    Any clashes — inspect structure manually


================================================================================
2. interface_remodeling.py — CDR MUTATION SCAN AND OPTIMISATION
================================================================================

Takes an already-validated, high-confidence VHH-antigen complex (e.g. a
structure that passed ensemble_pipeline.py's READY tier) and proposes point
mutations at its CDR residues, scoring each mutant with Boltz2. Intended as
the follow-up step for a promising hit from ensemble_pipeline.py, not a
replacement for it.

Design note — why it doesn't re-dock a fresh WT pose from scratch:
  Boltz2 diffusion is not guaranteed to converge on the correct pose, so
  deriving the reference epitope/binding mode independently here would risk
  propagating a bad pose into every mutant's score. Instead, the input
  --complex-cif is trusted as ground truth for the pose, and every mutant
  (including the WT itself, rescored as a baseline) is docked templated on
  it for BOTH chains, so all scores come from the same pose/conditions and
  are directly comparable. No fresh NanobodyBuilder2 run is needed per
  mutant, since a point mutant differs from the template by one residue and
  Boltz2's template alignment is sequence-based, not exact-match.

Pipeline:
  Stage 0   Load --complex-cif, extract chain A (binder)/chain B (antigen)
            sequences, ANARCI (IMGT scheme)-number the binder to locate
            CDR1/2/3 vs framework.
  Stage 1   Interface residues are read directly off --complex-cif (no
            docking) — used later for classification, not as a pre-filter.
  Stage 1b  WT rescoring: the WT sequence is docked under the exact same
            templated-on-both-chains conditions as every mutant. Its
            mean_binding_score becomes the baseline for binding_score_delta.
  Stage 2   Mutation generation: every residue in the selected CDR loop(s)
            is scanned to alanine (or glycine, if already alanine) — the
            whole CDR panel, not just residues already flagged as
            interface-contacting, so a position whose small WT sidechain
            doesn't currently reach the antigen is still tested.
  Stage 3   Each mutant is docked with Boltz2, templated on --complex-cif
            for both chains.
  Stage 4   Results are reported, ranked by binding_score_delta (ascending
            — biggest drops first).
  Stage 5   (--optimize only) Each round-1 position is classified as:
              hotspot          — interface-contacting, score drops well
                                 below the round-1 panel median; left alone
              tolerant_contact — interface-contacting, score barely moved
              cold_spot        — not currently interface-contacting at all
            A second round tries a panel of paratope-enriched substitutions
            (Tyr/Arg/Trp/Asp/Asn) at every tolerant_contact and cold_spot
            position, looking for a bulkier or more interactive sidechain
            that creates a new favourable contact — skipping confirmed
            hotspots.

Usage:
  python interface_remodeling.py --complex-cif validated/Cluster_12_model_1.cif
  python interface_remodeling.py --complex-cif validated/Cluster_12_model_1.cif --optimize

All flags and defaults:

REFERENCE STRUCTURE
  --complex-cif PATH
    Validated two-chain VHH-antigen complex (chain A = binder, chain B =
    antigen), e.g. a best/converged model from ensemble_pipeline.py's
    docking output. Required. Also the source of the WT sequence, which is
    rescored (Stage 1b) as the binding_score_delta baseline.

MUTATION SCAN
  --cdrs 1 2 3
    Which CDR loop(s) to scan.
    Default: 1 2 3 (all three)

  --optimize
    Run the round-2 paratope-enrichment panel (Stage 5) after the round-1
    Ala/Gly scan. Multiplies compute by roughly the number of qualifying
    (non-hotspot) positions × 5 substitutions.
    Default: off

  --hotspot-drop-fraction F
    A round-1 mutant scoring more than this fraction below the round-1
    panel median marks its position a hotspot (excluded from --optimize).
    Default: 0.075

DOCKING (same conventions as ensemble_pipeline.py)
  --num-models N            Diffusion samples per mutant docking run. Default: 5
  --recycling-steps N       Boltz2 recycling iterations per sample. Default: 3
  --max-parallel-samples N  Maximum diffusion samples run in parallel on the GPU.

HARDWARE AND MSA
  --accelerator gpu|cpu|tpu  Default: gpu
  --no-msa-server            Disable the ColabFold MSA server for the antigen chain.

OUTPUT
  --output DIR
    Output root directory.
    Default: a directory named after --complex-cif's own basename, in the cwd.

  --vis-dir DIR
    Directory collecting WT.cif (the reference complex) plus one CIF per
    mutant named after its mutation (e.g. I33A.cif), for loading together
    in a structure viewer (PyMOL/ChimeraX).
    Default: best_structures/ under --output.

Outputs (under --output, e.g. ./Cluster_12_model_1/):
  docking/{variant}/         Boltz2 rigid-body docking results per mutant,
                              plus the WT rescoring run itself.
  best_structures/            WT.cif (the input complex, copied in as-is) and
                              WT_rescored.cif (its best-scoring model from
                              Stage 1b) plus one CIF per mutant named after
                              its mutation, e.g. I33A.cif.
  mutation_candidates.csv    One row per mutant (round1, and round2 if
                              --optimize was used), ranked by
                              binding_score_delta against the Stage 1b WT
                              rescore (ascending — biggest drops first).
                              Round and Position_label (hotspot/
                              tolerant_contact/cold_spot, round2 rows only)
                              identify which stage/classification each row
                              came from.
  logs/                      Boltz2 stdout/stderr logs


================================================================================
3. manual_mutant_scan.py — SCORE YOUR OWN MUTANT SEQUENCES
================================================================================

Takes a FASTA of one or more hand-designed mutant VHH sequences and a
validated WT complex (--wt-cif), and docks each mutant templated on that
same complex for both chains — the same WT-pose-templated docking strategy
interface_remodeling.py uses for its automated CDR scan, applied here to
sequences you supply directly instead of an Ala/Gly or enhancement panel.
Use this when you want to test specific mutations (e.g. combinations, or
substitutions outside the standard scan panels) rather than running the
full automated scan.

The WT sequence (read from chain A of --wt-cif) is rescored under identical
conditions, giving a binding_score baseline so each mutant's
binding_score_delta is a like-for-like comparison.

Usage:
  python manual_mutant_scan.py --fasta my_mutants.fasta \
      --wt-cif validated/Cluster_12_model_1.cif
  python manual_mutant_scan.py --fasta my_mutants.fasta \
      --wt-cif validated/Cluster_12_model_1.cif \
      --num-models 10 --recycling-steps 5 --output my_mutants_out/

FASTA input:
  One or more mutant VHH sequences, e.g.:
    >I33A_S57Y
    QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMG...
  Each sequence is docked independently, templated on --wt-cif for both
  chains. The FASTA header becomes the variant's name in the report and
  output filenames.

Flags:
  --fasta PATH          FASTA file of mutant VHH sequences. Required.
  --wt-cif PATH         Validated WT complex (chain A = VHH, chain B =
                        antigen), used as the fixed pose template for the
                        WT rescore and every mutant. Required.
  --num-models N        Diffusion samples per docking run. Default: 5
  --recycling-steps N   Boltz2 recycling iterations per sample. Default: 5
                        (note: higher than interface_remodeling.py's
                        default of 3, deliberately, for this script)
  --max-parallel-samples N   Maximum diffusion samples run in parallel on the GPU.
  --accelerator gpu|cpu|tpu   Default: gpu
  --no-msa-server        Disable the ColabFold MSA server for the antigen chain.
  --output DIR           Output root directory.
                         Default: a directory named after --fasta's basename, in the cwd.

Outputs (under --output):
  docking/{variant}/        Boltz2 rigid-body docking results per mutant,
                             plus the WT rescoring run.
  best_structures/           WT.cif (the input --wt-cif, copied in as-is),
                             WT_rescored.cif (its best-scoring model from the
                             WT rescore), and one CIF per mutant named after
                             its FASTA header — load this directory directly
                             in a structure viewer to compare mutants
                             against the WT.
  scoring_report.txt         Human-readable summary: WT baseline, then one
                             block per mutant with binding_score,
                             binding_score_delta, ipTM, BSA, and clashes,
                             ranked by binding_score_delta (ascending —
                             biggest drops first).
  logs/                      Boltz2 stdout/stderr logs


================================================================================
4. biophysical_analysis.py — SEQUENCE BIOPHYSICAL PROPERTIES
================================================================================

Computes molecular weight, pI, GRAVY, aromaticity, aliphatic index, and
extinction coefficient (reduced/oxidized) for protein sequences from a
CSV/Excel file. No structure prediction — sequence-only, fast.

Usage:
  python biophysical_analysis.py input.csv
  python biophysical_analysis.py input.xlsx --column ProteinSeq
  python biophysical_analysis.py input.csv --output results.csv

Flags:
  input (positional)   CSV or Excel file. Required.
  --column COLUMN      Column containing protein sequences. Default: Sequence
  --output PATH        Output CSV path. Default: <input>_biophysical_results.csv
                        next to the input file.

Output: input columns plus Molecular Weight (Da), pI, GRAVY, Aromaticity,
Aliphatic Index, Ext. Coeff Reduced/Oxidized (M-1 cm-1) — one row per input
row, in the same order. Rows that fail to parse get blank result columns
rather than aborting the whole run.


================================================================================
5. nanobody_structure.py — STANDALONE NANOBODYBUILDER2 RUNNER
================================================================================

Runs ImmuneBuilder's NanoBodyBuilder2 on every sequence in a FASTA file,
producing all 4 OpenMM-refined conformers per sequence (not just rank-0).
Useful for visual inspection of the full 4-model spread, or as input
structures for a different downstream tool — does not run Boltz2 docking.

Usage:
  python nanobody_structure.py sequences.fasta
  python nanobody_structure.py sequences.fasta --output structures/
  python nanobody_structure.py sequences.fasta --no-refine

Flags:
  fasta (positional)   FASTA file of VHH sequences. Required.
  --output DIR         Output directory. Default: nb2_structures/ next to input.
  --no-refine          Skip OpenMM refinement (faster; structures may have
                        minor geometry issues).

Output per sequence (in --output/{name}/):
  {name}_model_1.pdb ... {name}_model_4.pdb   — all 4 refined conformers,
                                                 ranked (model_1 = best)
  {name}_best.pdb                              — copy of model_1


================================================================================
6. boltz_dock.py — LIGHTWEIGHT SINGLE-COMPLEX DOCKING
================================================================================

Simple Boltz2 rigid-body docking of one nanobody structure file against one
antigen structure file. Use this when you already have a VHH structure (from
nanobody_structure.py, a prior ensemble_pipeline.py run, or elsewhere) and
just want a quick docking run without the full NanobodyBuilder2 + convergence
scoring pipeline.

Usage:
  python boltz_dock.py --nanobody my_vhh.pdb --antigen antigen.pdb
  python boltz_dock.py --nanobody my_vhh.pdb --antigen Ab.pdb \
      --antigen-chains B D --output docking_results/
  python boltz_dock.py --nanobody my_vhh.pdb --antigen antigen.pdb \
      --models 3 --no-msa-server

Flags:
  --nanobody PATH        Nanobody/VHH structure (.pdb or .cif). Required.
  --antigen PATH         Antigen structure (.pdb or .cif). Required.
  --nanobody-chain ID    Chain ID to use from nanobody file. Default: first chain.
  --antigen-chain ID     Single antigen chain. Default: first chain.
                         (mutually exclusive with --antigen-chains)
  --antigen-chains ID [ID ...]
                         Multiple antigen chain IDs merged into one (e.g. Fc
                         homodimer). Residues concatenated and renumbered.
  --output DIR           Output directory. Default: boltz_docking/ next to antigen.
  --models N             Number of Boltz2 diffusion samples. Default: 3
  --max-parallel N       Max parallel GPU samples. Default: 1 (sequential, avoids OOM).
  --no-template          Do not provide structures as Boltz2 templates (free
                         docking, no structural constraints).
  --no-msa-server        Disable ColabFold MSA server (enabled by default for
                         both chains — unlike ensemble_pipeline.py, this
                         script has no single-sequence-mode exception for
                         the nanobody chain).
  --accelerator gpu|cpu|tpu   Default: gpu

Output: complex CIF structures + per-model binding_score/ipTM/pTM printed to
the terminal, written flat into --output/.


================================================================================
7. boltz_pipeline.py — BATCH SINGLE-CHAIN STRUCTURE PREDICTION
================================================================================

Bare Boltz2 structure prediction (no docking, no NanobodyBuilder2) for every
sequence in a CSV/Excel file. Use this for plain single-chain structure
prediction at scale — not VHH/antigen-specific.

Usage:
  python boltz_pipeline.py input.csv
  python boltz_pipeline.py input.xlsx --names ID --sequences Seq
  python boltz_pipeline.py input.csv --num-models 5 --accelerator gpu
  python boltz_pipeline.py input.csv --output results/ --output-format pdb

Flags:
  input (positional)     CSV or Excel file. Required.
  --names COLUMN         Sample/molecule name column. Default: Sample
  --sequences COLUMN     Protein sequence column. Default: Sequence
  --output DIR           Output directory. Default: boltz_predictions/ next to input.
  --accelerator gpu|cpu|tpu   Default: gpu
  --num-models N         Structure models per sequence. Default: 1
  --output-format mmcif|pdb   Default: mmcif
  --no-msa-server        Disable MSA server lookup (only if pre-computed MSAs provided).
  --no-monitor           Disable asitop/powermetrics usage monitoring during the run
                         (see monitor.py; macOS live dashboard + power log, or
                         Linux CPU/GPU polling — skipped automatically on other
                         platforms either way).

Output: {Sample}-1.cif, {Sample}-2.cif, etc. written flat into --output/,
plus a CPU/GPU usage plot (usage_<timestamp>.png) unless --no-monitor is set.


ANTIGEN STRUCTURES
------------------
  Fetch from AlphaFold DB (when no PDB structure exists):
    curl -o antigen.pdb \
      https://alphafold.ebi.ac.uk/files/AF-<UniProtID>-F1-model_v6.pdb

  Human CD7  (P09564): binding/h7/hCD7_alphafold.pdb  [already downloaded]

  Target-specific working data lives under binding/{target}/ (one subdirectory
  per antigen target, e.g. h7, h19, h20, h25, m2, m5, m7, m19, m20, hmsln,
  mmsln, hmuc1, mmuc1, tp1107). files/ holds active working inputs/outputs
  that don't yet belong to a specific target subdirectory (e.g. clone lists,
  in-progress remodeling runs).


DIRECTORY STRUCTURE
-------------------
  protein_analysis/
  ├── setup.sh                    <- Run this first on any new machine
  ├── requirements.txt
  ├── README.txt
  ├── ensemble_pipeline.py        <- Main pipeline (use this first)
  ├── interface_remodeling.py     <- Mutation scan/optimisation (use on hits)
  ├── manual_mutant_scan.py       <- Score your own hand-designed mutants
  ├── biophysical_analysis.py
  ├── nanobody_structure.py
  ├── boltz_dock.py
  ├── boltz_pipeline.py
  ├── monitor.py                  <- Imported by boltz_pipeline.py, not run directly
  ├── binding/
  │   └── {target}/                One subdirectory per antigen target
  │       ├── <antigen>.pdb/.cif
  │       └── <enriched_clusters>.csv
  └── files/                       Active working inputs/outputs

================================================================================
