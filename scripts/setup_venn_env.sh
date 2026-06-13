#!/bin/bash
# Setup script for Venn diagram visualization environment

set -e

echo "Setting up Venn diagram visualization environment..."

# Create venv if it doesn't exist
if [ ! -d "venv_venn" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_venn
fi

# Activate venv
source venv_venn/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "✓ Environment setup complete!"
echo ""
echo "To use this environment:"
echo "  source venv_venn/bin/activate"
