#!/bin/bash
# Bootstrap the protein analysis environment on any workstation.
# Creates a venv named 'protein' and installs all dependencies.
#
# Usage:
#   bash setup.sh
#   source protein/bin/activate

set -e

# Check Python version (3.10+ required for X | Y type hints)
python3 -c "
import sys
if sys.version_info < (3, 10):
    print(f'Error: Python 3.10+ required, found {sys.version}')
    sys.exit(1)
print(f'Python {sys.version.split()[0]} — OK')
"

# Create venv
if [ -d "protein" ]; then
    echo "Venv 'protein' already exists — skipping creation."
else
    python3 -m venv protein
    echo "Venv 'protein' created."
fi

# Install dependencies
echo "Installing dependencies..."
protein/bin/pip install --upgrade pip --quiet
protein/bin/pip install -r requirements.txt

echo ""
echo "Setup complete."
echo "Activate with:  source protein/bin/activate"
echo "Then run:       python boltz_pipeline.py --help"
