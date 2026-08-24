#!/usr/bin/env python3
"""
auto_temp.py — Automatic temperature adjuster for SmallThinker-3B.

Reads a prompt, classifies it, and outputs the empirically-best temperature
from our 20-run temperature sweep (results/temp_sweep_quality.json).

The sweep found that the meta-reasoning loop is CHAOTIC, not monotonic —
there is no single "best temperature." The best temp depends on what kind
of task the prompt is asking for:

  Classification          Best temp  Quality  Why
  ---------------------  ---------  -------  --------------------------------
  deep reasoning         1.0        8/10     Explores reasoning tree (math/proof)
  creative fiction       0.9        7/10     Only temp that commits to prose
  creative code          0.7        6/10     Closest to correct for games/retro
  expository prose       0.5        8/10     Robust; never loops
  strict-form poetry     0.3        3/10     Weakest task; only temp that emits

Usage:
  python3 auto_temp.py "your prompt text"
  echo "your prompt" | python3 auto_temp.py

Output (one line, space-separated): temperature  style  context  tokens
  e.g.  0.9  fiction  4096  1200

Exit code 0 on success, 1 on error.
"""
import sys
import re

# ── Keyword banks ──────────────────────────────────────────────────────
# Each category gets a list of keyword patterns. The classifier scores
# every category and picks the highest. Ties break toward the default
# (deep reasoning, temp 1.0) which is the safest for unknown prompts.

POETRY_WORDS = [
    r"\bpoem\b", r"\bsonnet\b", r"\bhaiku\b", r"\blimerick\b", r"\bode\b",
    r"\bvillanelle\b", r"\bsestina\b", r"\bacrostic\b", r"\bepigram\b",
    r"\biambic\b", r"\bpentameter\b", r"\bpentametre\b", r"\bblank verse\b",
    r"\brhyme\b", r"\brhyming\b", r"\bverse\b", r"\bstanza\b",
    r"\bmeter\b", r"\bmetre\b", r"\bscansion\b",
    r"write a poem", r"compose a poem", r"write (?:a |an )?(?:poem|sonnet|haiku)",
]

FICTION_WORDS = [
    r"\bstory\b", r"\bscene\b", r"\bnarrative\b", r"\bfiction\b",
    r"\btale\b", r"\bshort story\b", r"\bnovella\b",
    r"\bcharacter\b", r"\bdialogue\b", r"\bplot\b", r"\bprotagonist\b",
    r"\bantagonist\b", r"\bsetting\b.*\batmosphere\b",
    r"\bnarrator\b", r"\bfirst person\b", r"\bthird person\b",
    r"write a story", r"write a scene", r"tell (?:us |me )?(?:a |the )?(?:story|tale)",
    r"creative writing", r"write (?:a |an )?(?:opening |opening )?(?:scene|chapter)",
    r"lighthouse", r"stormy night", r"mysterious stranger",
]

PROSE_WORDS = [
    r"\bexplain\b", r"\bdescribe\b", r"\bsummarize\b", r"\boverview\b",
    r"\bdiscuss\b", r"\bcompare\b", r"\bcontrast\b",
    r"\bwhat is\b", r"\bwhat are\b", r"\bwhat causes\b",
    r"\bhow does\b", r"\bhow do\b", r"\bhow (?:does|do|did|can|could)\b",
    r"\bwrite about\b", r"\bdescribe the\b",
    r"\bpathophysiology\b", r"\bdiagnosis\b", r"\btreatment\b",
    r"\bclinical\b", r"\bmedical\b", r"\bdisease\b", r"\bcondition\b",
    r"\barticle\b", r"\bessay\b", r"\breport\b", r"\bbriefing\b",
]

# Creative code: games, interactive programs, retro computing —
# NOT algorithmic code (which falls through to deep reasoning at temp 1.0)
CREATIVE_CODE_WORDS = [
    r"\bgame\b", r"\bplayable\b", r"\bretro\b", r"\btext adventure\b",
    r"\binteractive\b", r"\bsimulation\b",
    r"\bBASIC\b", r"\bTRS-80\b", r"\bCommodore\b", r"\bApple II\b",
    r"\bsprite\b", r"\bscrolling\b", r"\bsound effect\b",
    r"write a game", r"write a text adventure", r"make a game",
]

# Deep reasoning: math, proofs, algorithms, logic —
# tasks with a CHECKABLE correct answer where exploration helps.
# This is the DEFAULT/fallback, so we only boost when we see strong signals
# that would otherwise be caught by another category.
DEEP_REASONING_WORDS = [
    r"\bprove\b", r"\bproof\b", r"\btheorem\b", r"\blemma\b", r"\bcorollary\b",
    r"\bsolve\b", r"\bcalculate\b", r"\bcompute\b", r"\bequation\b",
    r"\balgorithm\b", r"\bcomplexity\b", r"\bO\([nlog]+\)", r"\bbig-?O\b",
    r"\bstep by step\b", r"\bstep-by-step\b",
    r"\boptimize\b", r"\bdebug\b", r"\bfix\b.*\bbug\b",
    r"\bcontradiction\b", r"\binduction\b", r"\bderivation\b",
    r"\bmatrix\b", r"\beigen\b", r"\bintegral\b", r"\bderivative\b",
    r"\bsort\b", r"\bsearch\b", r"\btree\b.*\btraversal\b",
    r"\brecursion\b", r"\bdynamic programming\b",
    r"\bmerge sort\b", r"\bquick sort\b", r"\bbinary search\b",
    r"\bhash table\b", r"\blinked list\b", r"\bqueue\b.*\bstack\b",
]

# ── Scoring ───────────────────────────────────────────────────────────
def score_category(text: str, patterns: list[str]) -> int:
    """Count how many patterns match the prompt text (case-insensitive)."""
    count = 0
    t = text.lower()
    for pat in patterns:
        if re.search(pat, t):
            count += 1
    return count

def classify(prompt: str) -> tuple[float, str, int, int]:
    """
    Classify a prompt and return (temperature, style, context, tokens).

    Returns the empirically-best settings from the temperature sweep.
    Context and token budget are also adjusted per category:
      - Deep reasoning: max context + max tokens (the model needs room to think)
      - Creative tasks: moderate context + moderate tokens (prevents loop spiral)
    """
    scores = {
        "poetry":           score_category(prompt, POETRY_WORDS),
        "fiction":          score_category(prompt, FICTION_WORDS),
        "prose":            score_category(prompt, PROSE_WORDS),
        "creative_code":    score_category(prompt, CREATIVE_CODE_WORDS),
        "deep_reasoning":   score_category(prompt, DEEP_REASONING_WORDS),
    }

    # Pick the highest-scoring category.
    # Ties break toward deep_reasoning (the safe default at temp 1.0).
    best = max(scores, key=lambda k: (scores[k], k == "deep_reasoning"))

    # If nothing matched at all, default to deep reasoning.
    if scores[best] == 0:
        best = "deep_reasoning"

    # Map category → (temp, style, context, tokens)
    settings = {
        "poetry":         (0.3, "poetry",         4096, 1200),
        "fiction":        (0.9, "fiction",         4096, 1200),
        "prose":          (0.5, "prose",           4096, 2048),
        "creative_code":  (0.7, "code",            4096, 2048),
        "deep_reasoning": (1.0, "reasoning",      32768, 16384),
    }

    return settings[best]

# ── Main ──────────────────────────────────────────────────────────────
def main():
    # Read prompt from arg or stdin
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        print("Usage: python3 auto_temp.py 'your prompt'", file=sys.stderr)
        sys.exit(1)

    if not prompt:
        print("Error: empty prompt", file=sys.stderr)
        sys.exit(1)

    temp, style, ctx, tokens = classify(prompt)

    # Output: temp style context tokens (space-separated, for shell parsing)
    print(f"{temp} {style} {ctx} {tokens}")

if __name__ == "__main__":
    main()