#!/usr/bin/env bash
# deep-smallthinker-3b — single-turn deep reasoning on an 8GB Jetson.
#
# The whole point: give SmallThinker-3B an EXCESS of thinking tokens on a
# single turn, with a high temperature (we have free tokens, so we let it
# explore the reasoning tree) and a context sized for GUI-off operation.
#
# Usage:
#   ./think.sh "your question here"
#   PROMPT_FILE=/path/to/prompt.txt ./think.sh
#
# GUI mode (env GUI_MODE, default "max"):
#   max  -> context 32768, thinking tokens 16384  (native 32K; GUI MUST be off)
#   off  -> context 16384, thinking tokens 8192   (GUI off, lighter RAM)
#   on   -> context  8192, thinking tokens 4096   (GUI still running)
set -euo pipefail

# --- Paths ---
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/smallthinker-3b-q4_k_m.gguf}"

# --- GUI mode -> context + thinking budget ---
GUI_MODE="${GUI_MODE:-max}"
case "$GUI_MODE" in
  off) CTX="${CTX:-16384}";  N_TOKENS="${N_TOKENS:-8192}" ;;
  on)  CTX="${CTX:-8192}";   N_TOKENS="${N_TOKENS:-4096}" ;;
  max) CTX="${CTX:-32768}";  N_TOKENS="${N_TOKENS:-16384}" ;;
  *) echo "GUI_MODE must be one of: off | on | max" >&2; exit 1 ;;
esac

# --- Sampling: high temperature because we have free tokens ---
# Temp 1.0 (the paper's recommendation) maximizes exploration of the
# reasoning tree. Drop TEMP to 0.6 for math/code where precision > breadth.
TEMP="${TEMP:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-40}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.1}"

# --- Prompt: arg, file, or stdin ---
if [[ $# -gt 0 ]]; then
  PROMPT="$*"
elif [[ -n "${PROMPT_FILE:-}" ]]; then
  PROMPT="$(cat "$PROMPT_FILE")"
elif [[ ! -t 0 ]]; then
  PROMPT="$(cat)"
else
  echo "Usage: $0 'your question here'" >&2
  echo "   or: PROMPT_FILE=/path/prompt.txt $0" >&2
  echo "   or: echo 'question' | $0" >&2
  exit 1
fi

# --- Show effective settings (transparency) ---
echo "== deep-smallthinker-3b ==" >&2
echo "  model:     $MODEL" >&2
echo "  gui_mode:  $GUI_MODE" >&2
echo "  context:   $CTX" >&2
echo "  think_tok: $N_TOKENS" >&2
echo "  temp:      $TEMP (top_p=$TOP_P top_k=$TOP_K rep=$REPEAT_PENALTY)" >&2
echo "===========================" >&2

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
