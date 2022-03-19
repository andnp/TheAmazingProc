#!/bin/bash
set -e

export PYTHONPATH=TheAmazingProc
python3 -m unittest discover -p "*test_*.py"
