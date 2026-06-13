================================================================================
 PROTEIN ANALYSIS PIPELINE
================================================================================

REQUIREMENTS
------------
- Python 3.10 or newer
- Internet access (for MSA server during Boltz predictions)
- GPU strongly recommended (CUDA on Linux/Windows)
- First run downloads ~3 GB of Boltz2 model weights to ~/.boltz (cached)
- First run downloads ImmuneBuilder model weights (~500 MB, cached)


SETUP (run once on any new workstation)
----------------------------------------
  bash setup.sh

This creates a virtual environment called 'protein/' and installs all
dependencies including ImmuneBuilder (NanobodyBuilder2) and Boltz2.
Do NOT copy the 'protein/' folder between machines — always run setup.sh
on the target machine instead.

Activate the environment before running any script:
  source protein/bin/activate          # macOS / Linux
  protein\Scripts\activate             # Windows


SCRIPTS
-------

1. biophysical_analysis.py
   Calculates biophysical properties for protein sequences from CSV/Excel.
   Outputs: molecular weight, pI, GRAVY, aromaticity, aliphatic index,
            extinction coefficient.

   Usage:
     python biophysical_analysis.py input.xlsx
     python biophysical_analysis.py input.csv --column MySeqColumn

   Output: <input>_biophysical_results.csv (same directory as input)


2. ensemble_pipeline.py
   VHH–antigen rigid-body docking pipeline for enriched phage display cluster
   representatives. Takes a CSV of VHH sequences (with enrichment scores) and
   an antigen structure, and produces convergence-scored complex predictions.

   Background:
     VHH sequences entering this pipeline are cluster representatives selected
     by Levenshtein distance (0.85 threshold) from ONT long-read sequencing of
     a phage display experiment, filtered for log2-fold enrichment between
     panning rounds. The pipeline is the final computational triage step before
     wet-lab ordering.

     NanobodyBuilder2 generates 4 structures from 4 independently pre-trained
     networks. The rank-0 (highest-confidence) structure is OpenMM-relaxed and
     used as the VHH template for all Boltz2 docking. Templating both chains
     (VHH + antigen) implements rigid-body docking: the VHH conformation is
     fixed, and the diffusion process samples interface geometry. Convergence
     across diffusion samples is the primary confidence signal.

   Three automated stages:
     Stage 1  NanobodyBuilder2 (ImmuneBuilder) — generates 4 model-diverse VHH
              structures; selects and OpenMM-relaxes the rank-0 (best) model.
     Stage 2  Rigid-body Boltz2 docking — runs a full two-chain (VHH + antigen)
              complex prediction using the rank-0 VHH as a fixed structural
              template for chain A (recycling_steps=3). --num-models diffusion
              samples are generated in a single Boltz2 call (default: 5).
     Stage 3  Convergence scoring — computes epitope overlap (fraction of antigen
              residues contacted by ≥50% of diffusion samples) and pose RMSD
              across samples. convergence_rank = epitope_overlap ×
              mean_binding_score is the primary output metric.

   Input CSV (one row per cluster representative):
     Cluster_R2           — cluster name / ID      (--names)
     Protein_Sequence_R2  — VHH amino acid sequence (--sequences)
     Log2_Enrichment      — panning enrichment score (--enrichment, optional)

   Typical usage:
     python ensemble_pipeline.py h7/h7_R1vsR2_known_enriched.csv \
       --antigen h7/hCD7_alphafold.pdb \
       --names Cluster_R2 \
       --sequences Protein_Sequence_R2 \
       --use-template

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

   STAGE 2 — RIGID-BODY BOLTZ2 DOCKING
     --num-models N
       Number of Boltz2 diffusion samples for the rigid-body docking run. All
       samples use the rank-0 OpenMM-relaxed VHH as a fixed structural template.
       Mean ± std are reported across samples.
       recycling_steps is fixed at 3.
       Default: 5

     --max-parallel-samples N
       Maximum number of diffusion samples to run simultaneously on the GPU.
       Reduce if CUDA runs out of memory (OOM errors).
       Default: equal to --num-models (all samples in parallel).

   HARDWARE AND MSA
     --accelerator gpu|cpu|tpu
       Hardware accelerator passed to Boltz2.
       Default: gpu

     --no-msa-server
       Disable the ColabFold MSA server. Use only when pre-computed MSAs are
       available in the expected location, or for fully offline runs.
       MSA fetching may be rate-limited on the public server; this is normal.
       Default: off (MSA server enabled)

   Outputs (under --output/):
     vhh_structures/{name}/
       {name}_best_model.pdb          — rank-0 NanobodyBuilder2 structure,
                                        OpenMM-relaxed (used as docking template)
       {name}_best_model.cif          — CIF conversion for Boltz2

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

   Interpretation:
     Strong ordering candidate:
       High convergence_rank (e.g. >0.5) + high Log2_Enrichment + tier READY
     Uncertain / do not order without further validation:
       Low epitope_overlap_fraction (<0.3) even with high mean_binding_score
       High pose_convergence_rmsd (>5 Å) — poses diverge across diffusion samples
       Any clashes — inspect structure manually


ANTIGEN STRUCTURES
------------------
  Fetch from AlphaFold DB (when no PDB structure exists):
    curl -o antigen.pdb \
      https://alphafold.ebi.ac.uk/files/AF-<UniProtID>-F1-model_v6.pdb

  Human CD7  (P09564): h7/hCD7_alphafold.pdb  [already downloaded]


DIRECTORY STRUCTURE
-------------------
  protein_analysis/
  ├── setup.sh                  <- Run this first on any new machine
  ├── requirements.txt
  ├── README.txt
  ├── biophysical_analysis.py
  ├── ensemble_pipeline.py      <- Main pipeline (use this)
  └── h7/
      ├── hCD7_alphafold.pdb
      └── h7_R1vsR2_known_enriched.csv

================================================================================
