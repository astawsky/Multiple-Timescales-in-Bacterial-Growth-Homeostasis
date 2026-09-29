#!/usr/bin/env bash
# Run from the repo root:
#   bash analysis/pymc_marketing/run.sh

set -e

PYTHON=/opt/homebrew/Caskroom/miniforge/base/envs/pymc-marketing/bin/python
DIR=analysis/pymc_marketing

echo "=== Stage A: fit single-chain MMM ==="
$PYTHON $DIR/test_single_chain.py

echo ""
echo "=== Evaluation: accuracy + length reconstruction ==="
$PYTHON $DIR/evaluate.py
