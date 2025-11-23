#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/mnt/data/af2_inputs"
OUT_DIR="/mnt/data/results_af2"
mkdir -p "$OUT_DIR"

# Run AlphaFold baseline via ColabFold batch. Requires colabfold_batch in PATH.
# We use single_sequence MSA mode to match ESMFold (no MSA). Adjust if you want MSAs.
colabfold_batch --msa-mode single_sequence "$INPUT_DIR" "$OUT_DIR"

# Ensure files are named <id>.pdb (colabfold usually embeds the input basename already).
# If outputs are in subfolders, move them up. This block is safe to run repeatedly.
shopt -s globstar nullglob
for f in "$OUT_DIR"/**/*.pdb; do
  base="$(basename "$f")"
  # move into OUT_DIR root
  if [[ "$f" != "$OUT_DIR/$base" ]]; then
    mv -f "$f" "$OUT_DIR/$base"
  fi
done

echo "[done] ColabFold AF2 baselines in: $OUT_DIR"
