#!/bin/bash
set -e
echo "Installing base packages..."
python -m pip install --upgrade pip==24.0
pip install setuptools==69.5.1 wheel==0.43.0
echo "Installing requirements..."
pip install --no-cache-dir -r requirements.txt