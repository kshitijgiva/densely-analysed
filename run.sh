#!/bin/bash

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary folders
mkdir -p data/raw
mkdir -p results

# Run main script
echo "Starting CCTV Analytics..."
python src/main.py

deactivate
