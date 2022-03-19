#!/bin/bash
set -e

MYPYPATH=./typings mypy -p AmazingProc

export PYTHONPATH=AmazingProc
python3 -m unittest discover -p "*test_*.py"
