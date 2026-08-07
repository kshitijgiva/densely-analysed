#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Create virtual environment
python -m venv venv
if [ -f venv/Scripts/activate ]; then
    source venv/Scripts/activate   # Windows venv layout
else
    source venv/bin/activate       # macOS/Linux venv layout
fi

# Install dependencies
pip install -r analytics_service/src/requirements.txt

# Create necessary folders
mkdir -p data/raw
mkdir -p analytics_service/results

# Run the M1/M2 validation pipeline (detection + tracking + re-id) against
# data/raw/sample_video.mp4. Use realtime.py instead for the live cv2.imshow demo.
echo "Starting CCTV Analytics validation pipeline..."
cd analytics_service/src
python validate_pipeline.py "$@"

deactivate
