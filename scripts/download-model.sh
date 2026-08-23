#!/usr/bin/env bash
# Download the Q4_K_M GGUF of SmallThinker-3B (2.0 GB) for the 8GB Jetson.
set -euo pipefail

URL="https://huggingface.co/QuantFactory/SmallThinker-3B-Preview-GGUF/resolve/main/SmallThinker-3B-Preview.Q4_K_M.gguf"
DEST="${1:-$HOME/models/smallthinker-3b-q4_k_m.gguf}"

mkdir -p "$(dirname "$DEST")"

if [[ -f "$DEST" ]] && [[ $(stat -c%s "$DEST" 2>/dev/null || echo 0) -gt 1900000000 ]]; then
  echo "Already present: $DEST"
  exit 0
fi

echo "Downloading SmallThinker-3B Q4_K_M (~2.0 GB) ..."
curl -L --fail --progress-bar "$URL" -o "$DEST"

echo "Done. Verify with:"
echo "  llama-cli -m $DEST -p 'hi' -n 16 -ngl 99"
