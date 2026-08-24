#!/usr/bin/env bash
# think-creative.sh — creative-form prompts for SmallThinker-3B Q4_K_M.
#
# Unlike think.sh (which defaults to temp 1.0 / max tokens for DEEP REASONING),
# this script is for OPEN-ENDED CREATIVE output where deep thinking is harmful.
#
# WHY: A 20-run temperature sweep (temp 0.1-0.9, everything else fixed) found
# that SmallThinker falls into a meta-reasoning loop ("I need to write X...
# first I should... let me think...") on creative prompts at most temperatures,
# and there is NO single safe temperature — the loop is chaotic, and the best
# temperature depends on the kind of creative task:
#
#   Task type              Best temp   Quality   Finding
#   ---------------------  ----------  --------  -----------------------------
#   creative fiction       0.9         7/10      Only temp that commits to prose
#   expository prose       0.5-0.9     8/10      Robust at every temp (no loop)
#   strict-form poetry     0.3         3/10      Weakest; wrong meter/rhyme even at best
#   code / structured      0.7         6/10      Closest to correct output
#
# So this script takes a --style flag and picks the empirically-best
# temperature for that style. Default is "fiction" (the strongest creative
# result of the sweep).
#
# Usage:
#   ./think-creative.sh "Write a scene..."                    # fiction (temp 0.9)
#   ./think-creative.sh --style poetry "Write a sonnet..."    # temp 0.3
#   ./think-creative.sh --style prose "Explain..."            # temp 0.5
#   ./think-creative.sh --style code "Write a program..."     # temp 0.7
#
# Full data: results/temp_sweep_quality.json
set -euo pipefail

# --- Paths ---
LLAMA_CLI="${LLAMA_CLI:-$HOME/llama.cpp/build/bin/llama-cli}"
MODEL="${MODEL:-$HOME/models/smallthinker-3b-q4_k_m.gguf}"

# --- Style -> temperature (from the sweep) ---
STYLE="fiction"
if [[ "${1:-}" == "--style" ]]; then
    STYLE="${2:-fiction}"
    shift 2
fi

case "$STYLE" in
    fiction)  TEMP="0.9" ;;
    prose)    TEMP="0.5" ;;
    poetry)   TEMP="0.3" ;;
    code)     TEMP="0.7" ;;
    *) echo "Unknown style '$STYLE'. Use: fiction | prose | poetry | code" >&2; exit 1 ;;
esac

# Creative tasks don't need 32K context or 8K+ thinking tokens. A moderate
# budget prevents the model from spiraling into endless re-planning.
CTX="${CTX:-4096}"
N_TOKENS="${N_TOKENS:-1200}"

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
    echo "Usage: $0 [--style fiction|prose|poetry|code] 'your prompt'" >&2
    exit 1
fi

echo "== think-creative.sh ==" >&2
echo "  style:     $STYLE -> temp $TEMP (empirically best from temp sweep)" >&2
echo "  context:   $CTX | tokens: $N_TOKENS" >&2
echo "=========================" >&2

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
