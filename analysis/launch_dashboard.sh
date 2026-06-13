#!/bin/bash

# Launch script for YOLO Metamorphic Testing Dashboard

echo "🎯 Launching YOLO Metamorphic Testing Dashboard..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "❌ results/ directory not found"
    echo "   Make sure you're running this from the analysis/ directory"
    exit 1
fi

# Launch the server
python3 serve_dashboard.py
