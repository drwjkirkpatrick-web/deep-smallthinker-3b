#!/usr/bin/env bash
# think-auto.sh — SmallThinker-3B with AUTOMATIC temperature adjustment.
#
# Reads your prompt, classifies it (reasoning / fiction / poetry / prose / code),
# and picks the empirically-best temperature from a 20-run temperature sweep.
# Everything else (context, tokens, sampling) is also auto-sized per category.
#
# This is the "no knobs" launcher — just give it a prompt and it figures out
# the best settings. For manual control, use think.sh or think-creative.sh.
#
# Usage:
#   ./think-auto.sh "your question or prompt here"
#   echo "your prompt" | ./think-auto.sh
#
# How it works:
#   1. auto_temp.py reads the prompt and outputs: temp style context tokens
#   2. This script parses that and passes it to llama-cli
#   3. You get the best result without guessing temperatures
set -euo pipefail

# --- Paths ---
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/smallthinker-3b-q4_k_m.gguf}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Prompt: arg, file, or stdin ---
if [[ $# -gt 0 ]]; then
  PROMPT="$*"
elif [[ -n "${PROMPT_FILE:-}" ]]; then
  PROMPT="$(cat "$PROMPT_FILE")"
elif [[ ! -t 0 ]]; then
  PROMPT="$(cat)"
else
  echo "Usage: $0 'your question or prompt'" >&2
  echo "   or: echo 'prompt' | $0" >&2
  echo "   or: PROMPT_FILE=/path/prompt.txt $0" >&2
  exit 1
fi

if [[ -z "$PROMPT" ]]; then
  echo "Error: empty prompt" >&2
  exit 1
fi

# --- Auto-classify the prompt ---
# auto_temp.py outputs: "temp style context tokens" (space-separated)
AUTO_OUTPUT=$(python3 "$SCRIPT_DIR/auto_temp.py" "$PROMPT" 2>/dev/null || echo "")
if [[ -z "$AUTO_OUTPUT" ]]; then
  echo "Warning: auto_temp.py failed, falling back to deep reasoning (temp 1.0)" >&2
  TEMP="1.0"; STYLE="reasoning"; CTX="32768"; N_TOKENS="16384"
else
  read -r TEMP STYLE CTX N_TOKENS <<< "$AUTO_OUTPUT"
fi

# --- Sampling (temperature is the auto-adjusted variable; rest is fixed) ---
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-40}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.1}"

# --- Show what the auto-adjuster picked ---
echo "== think-auto.sh (auto temperature) ==" >&2
echo "  prompt:    ${PROMPT:0:80}..." >&2
echo "  style:     $STYLE" >&2
echo "  temp:      $TEMP  ← auto-selected" >&2
echo "  context:   $CTX" >&2
echo "  tokens:    $N_TOKENS" >&2
echo "  top_p:     $TOP_P  |  top_k: $TOP_K  |  repeat: $REPEAT_PENALTY" >&2
echo "========================================" >&2

exec "$LLAMA_CLI" \
  -m "$MODEL" \
  -p "$PROMPT" \
  -n "$N_TOKENS" \
  -c "$CTX" \
  --temp "$TEMP" \
  --top-p "$TOP_P" \
  --top-k "$TOP_K" \
  --repeat-penalty "$REPEAT_PENALTY" \
  -t 6 \
  -ngl 99 \
  -fa on \
  --jinja \
  --no-conversation \
  --no-display-prompt \
  -st