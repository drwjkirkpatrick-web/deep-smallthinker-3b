#!/usr/bin/env bash
# think-auto.sh — SmallThinker-3B with AUTOMATIC settings adjustment.
#
# Reads your prompt, classifies it (reasoning / fiction / poetry / prose / code),
# and picks the empirically-best temperature AND sampling parameters from
# our two test sweeps (20-run temperature sweep + 45-run variable sweep).
# Everything (context, tokens, top_p, top_k, repeat_penalty) is auto-sized.
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
#   2. This script maps the style to tuned sampling params (top_p, top_k, rep)
#   3. llama-cli runs with all tuned settings
#
# Tuned defaults per category (from 45-run variable sweep):
#   reasoning:  top_p=0.80 top_k=80  rep=1.10  (fast + complete proofs)
#   fiction:    top_p=0.90 top_k=80  rep=1.20  (fast, no loop, clean output)
#   poetry:     top_p=0.90 top_k=80  rep=1.20  (same as fiction)
#   prose:      top_p=0.90 top_k=80  rep=1.20  (same as fiction)
#   code:       top_p=1.00 top_k=40  rep=1.20  (concise, fast, complete)
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
  echo "Warning: auto_temp.py failed, falling back to deep reasoning" >&2
  TEMP="1.0"; STYLE="reasoning"; CTX="32768"; N_TOKENS="16384"
else
  read -r TEMP STYLE CTX N_TOKENS <<< "$AUTO_OUTPUT"
fi

# --- Tuned sampling params per style (from 45-run variable sweep) ---
# Each style gets its own top_p, top_k, repeat_penalty optimized for it.
case "$STYLE" in
  reasoning)
    TOP_P="${TOP_P:-0.80}"
    TOP_K="${TOP_K:-80}"
    REPEAT_PENALTY="${REPEAT_PENALTY:-1.10}"
    ;;
  fiction|poetry|prose)
    TOP_P="${TOP_P:-0.90}"
    TOP_K="${TOP_K:-80}"
    REPEAT_PENALTY="${REPEAT_PENALTY:-1.20}"
    ;;
  code)
    TOP_P="${TOP_P:-1.00}"
    TOP_K="${TOP_K:-40}"
    REPEAT_PENALTY="${REPEAT_PENALTY:-1.20}"
    ;;
  *)
    TOP_P="${TOP_P:-0.80}"
    TOP_K="${TOP_K:-80}"
    REPEAT_PENALTY="${REPEAT_PENALTY:-1.10}"
    ;;
esac

# --- Show what the auto-adjuster picked ---
echo "== think-auto.sh (auto-tuned) ==" >&2
echo "  prompt:    ${PROMPT:0:80}..." >&2
echo "  style:     $STYLE" >&2
echo "  temp:      $TEMP  (auto-selected)" >&2
echo "  context:   $CTX" >&2
echo "  tokens:    $N_TOKENS" >&2
echo "  top_p:     $TOP_P  |  top_k: $TOP_K  |  repeat: $REPEAT_PENALTY" >&2
echo "  [tuned via 45-run variable sweep]" >&2
echo "=================================" >&2

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