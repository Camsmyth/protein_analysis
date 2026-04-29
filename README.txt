================================================================================
 PROTEIN ANALYSIS PIPELINE
================================================================================

REQUIREMENTS
------------
- Python 3.10 or newer
- Internet access (for MSA server during Boltz predictions)
- GPU recommended (MPS on Apple Silicon, CUDA on Linux/Windows)


SETUP (run once on any new workstation)
----------------------------------------
  bash setup.sh

This creates a virtual environment called 'protein/' and installs all
dependencies. Do NOT copy the 'protein/' folder between machines — always
run setup.sh on the target machine instead.

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


2. boltz_pipeline.py
   Runs Boltz2 structure prediction for sequences from CSV/Excel.
   Outputs one .cif structure file per sequence named {Sample}-1.cif.

   Usage:
     python boltz_pipeline.py input.xlsx
     python boltz_pipeline.py input.xlsx --num-models 3 --output my_structures/

   Flags:
     --names COLUMN        Column with sample names (default: Sample)
     --sequences COLUMN    Column with sequences (default: Sequence)
     --output DIR          Output directory (default: boltz_predictions/ next to input)
     --accelerator         gpu / cpu / tpu (default: gpu)
     --num-models N        Number of structures per sequence (default: 1)
     --no-msa-server       Disable MSA lookup (only if pre-computed MSAs provided)
     --no-monitor          Disable asitop / powermetrics monitoring

   Notes:
     - First run downloads ~3 GB of model weights to ~/.boltz (cached for future runs)
     - MSA fetching from the public ColabFold server may be rate-limited; this is normal
     - On macOS: run 'sudo -v' before starting to enable GPU/power monitoring
     - Structures are saved to boltz_predictions/raw/ during the run and renamed
       after completion — safe if the run is interrupted


3. binding_prediction.py
   Predicts binding of binder sequences against an antigen structure using
   Boltz2, and scores each complex with BindCraft/AF2-style metrics.

   Usage:
     python binding_prediction.py binders.xlsx --antigen antigen.pdb
     python binding_prediction.py binders.xlsx --antigen antigen.pdb \
       --binder-structures boltz_predictions/hCD7/ --use-template

   Flags:
     --antigen PATH            Antigen structure file (.pdb or .cif) [required]
     --binder-structures DIR   Directory of predicted binder .cif files.
                               Files must be named {Sample}-1.cif.
                               Uses existing structures as chain A templates
                               instead of re-predicting from sequence.
     --antigen-chain ID        Chain to use from antigen file (default: first)
     --use-template            Use antigen structure as Boltz2 template (chain B).
                               Recommended when antigen is from experiment (PDB).
                               Skip if antigen is itself an AF2/Boltz prediction.
     --output DIR              Output directory (default: binding_predictions/)
     --accelerator             gpu / cpu / tpu (default: gpu)
     --num-models N            Diffusion samples per binder (default: 1)

   Output: binding_predictions/binding_scores.csv with per-binder metrics:
     binding_score     — 0.8*ipTM + 0.2*pTM (primary ranking, higher = better)
     iptm              — interface TM-score (>0.7 = strong binding)
     ptm               — overall fold quality
     complex_plddt     — mean confidence across the complex
     complex_iplddt    — confidence at the interface
     pae_interface     — mean cross-chain PAE (lower = better)
     pae_binder        — intra-binder PAE
     pae_antigen       — intra-antigen PAE
     bsa_A2            — buried surface area (Angstrom^2)
     interface_contacts — number of residue pairs within 5 Angstrom (Ca-Ca)
     Results are written incrementally — safe if the run is interrupted.


ANTIGEN STRUCTURES
------------------
  Fetch from AlphaFold DB (when no PDB structure exists):
    curl -o antigen.pdb \
      https://alphafold.ebi.ac.uk/files/AF-<UniProtID>-F1-model_v6.pdb

  Human CD7  (P09564): antigens/hCD7_alphafold.pdb  [already downloaded]


MONITORING (macOS only)
-----------------------
  boltz_pipeline.py automatically launches asitop (live GPU/CPU dashboard)
  and logs power usage during runs. To enable powermetrics logging:
    sudo -v      # cache sudo credentials before starting the pipeline

  Usage plot is saved to the output directory after each run.
  Use --no-monitor to disable all monitoring.

  On Linux: CPU usage is polled from /proc/stat and GPU from nvidia-smi
  (if available). Same plot is generated automatically.


DIRECTORY STRUCTURE
-------------------
  protein_analysis/
  ├── setup.sh                  <- Run this first on any new machine
  ├── requirements.txt
  ├── README.txt
  ├── biophysical_analysis.py
  ├── boltz_pipeline.py
  ├── binding_prediction.py
  ├── monitor.py
  └── antigens/
      └── hCD7_alphafold.pdb

================================================================================
