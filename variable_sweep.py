#!/usr/bin/env python3
"""
Variable sweep for SmallThinker-3B Q4_K_M — organized fine-tuning tests.

Tests ONE variable at a time, holding everything else fixed at known-best
values. Context is ALWAYS 32768 (per user instruction). Temperature uses
the per-category best from the temperature sweep.

Variables tested:
  1. top_p:           0.80, 0.90, 0.95, 0.98, 1.0    (5 values)
  2. top_k:            0,   20,   40,   60,   80      (5 values)
  3. repeat_penalty:   1.0,  1.05, 1.1,  1.15, 1.2   (5 values)

Prompts (3 representative, one per category):
  reasoning:  "Prove that sqrt(2) is irrational."        (temp 1.0)
  creative:   "Write a short scene where a lighthouse    (temp 0.9)
              keeper meets a mysterious stranger
              on a foggy night."
  coding:     "Implement merge sort in Python with        (temp 1.0)
              type hints and a docstring."

Total: 3 prompts × 5 values × 3 variables = 45 runs.
Token budget: 4096 (enough for a complete answer, fast enough for 45 runs).
Context: 32768 (fixed per user instruction).
"""
import json
import os
import re
import subprocess
import sys
import time

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
MODEL = os.path.expanduser("~/models/smallthinker-3b-q4_k_m.gguf")
CTX = 32768          # FIXED — user instruction
N_TOKENS = 4096      # enough for a complete answer, keeps 45 runs manageable
THREADS = 6

# ── Prompts with their best temperatures ──────────────────────────────
PROMPTS = [
    {
        "id": "reasoning",
        "label": "Math Proof (sqrt(2) irrational)",
        "text": "Prove that sqrt(2) is irrational.",
        "temp": 1.0,
    },
    {
        "id": "creative",
        "label": "Creative Fiction (lighthouse scene)",
        "text": "Write a short scene where a lighthouse keeper meets a mysterious stranger on a foggy night.",
        "temp": 0.9,
    },
    {
        "id": "coding",
        "label": "Python Coding (merge sort)",
        "text": "Implement merge sort in Python with type hints and a docstring. Include edge case handling.",
        "temp": 1.0,
    },
]

# ── Variable definitions ──────────────────────────────────────────────
# Each sweep tests ONE variable. All others stay at defaults.
DEFAULTS = {
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
}

SWEEPS = [
    {
        "variable": "top_p",
        "label": "Nucleus sampling threshold (top_p)",
        "values": [0.80, 0.90, 0.95, 0.98, 1.0],
        "description": "Controls how many of the most-likely tokens are considered. Lower = more focused (only the very likely tokens). Higher = more diverse (more tokens in the candidate pool).",
    },
    {
        "variable": "top_k",
        "label": "Top-K sampling (top_k)",
        "values": [0, 20, 40, 60, 80],
        "description": "Limits to the K most-likely tokens. 0 = disabled (let top_p handle it). Lower K = more focused. Higher K = more diverse.",
    },
    {
        "variable": "repeat_penalty",
        "label": "Repetition penalty (repeat_penalty)",
        "values": [1.0, 1.05, 1.1, 1.15, 1.2],
        "description": "Penalizes tokens that already appeared. 1.0 = no penalty. Higher = more penalty, discourages repetition. Too high = model avoids common words and produces gibberish.",
    },
]

# ── Output parsing ─────────────────────────────────────────────────────
def parse_output(text: str) -> dict:
    """Extract metrics from llama-cli stderr + stdout."""
    gen_tps = 0.0
    prompt_tps = 0.0
    wall_time = 0.0

    # llama-cli prints timing to stderr; our subprocess captures both
    m = re.search(r"Generation:\s*([\d.]+)\s*t/s", text)
    if m:
        gen_tps = float(m.group(1))
    m = re.search(r"Prompt:\s*([\d.]+)\s*t/s", text)
    if m:
        prompt_tps = float(m.group(1))

    # Thinking vs answer split
    thinking = ""
    answer = ""
    # SmallThinker uses <think>...</think> blocks (via --jinja)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    else:
        # No <think> tags — check for "Final Answer" or similar
        answer_match = re.search(r"(?:Final Answer|Answer:|The answer is|\\boxed)", text)
        if answer_match:
            thinking = text[:answer_match.start()].strip()
            answer = text[answer_match.start():].strip()
        else:
            # If no delimiter, everything is "answer" (prose-style)
            answer = text.strip()

    think_chars = len(thinking)
    ans_chars = len(answer)
    total_chars = think_chars + ans_chars
    think_pct = (think_chars / total_chars * 100) if total_chars > 0 else 0

    # Truncation: did we hit the token limit?
    truncated = "n_eval" in text and gen_tps > 0 and ans_chars < 50 and think_chars > 1000

    return {
        "gen_tps": gen_tps,
        "prompt_tps": prompt_tps,
        "think_chars": think_chars,
        "ans_chars": ans_chars,
        "think_pct": round(think_pct, 1),
        "truncated": truncated,
        "thinking": thinking[:500],  # first 500 chars for debugging
        "answer": answer[:2000],      # first 2000 chars for scoring
    }

# ── Run one llama-cli invocation ───────────────────────────────────────
def run_one(prompt_text, temp, top_p, top_k, repeat_penalty, label=""):
    cmd = [
        LLAMA_CLI,
        "-m", MODEL,
        "-p", prompt_text,
        "-n", str(N_TOKENS),
        "-c", str(CTX),
        "--temp", str(temp),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--repeat-penalty", str(repeat_penalty),
        "-t", str(THREADS),
        "-ngl", "99",
        "-fa", "on",
        "--jinja",
        "--no-conversation",
        "--no-display-prompt",
        "-st",
    ]

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max per run
    )
    wall = time.time() - t0

    combined = result.stdout + result.stderr
    metrics = parse_output(combined)
    metrics["wall_time"] = round(wall, 1)
    metrics["full_output"] = result.stdout[:10000]  # for quality scoring later
    return metrics

# ── Main sweep ─────────────────────────────────────────────────────────
def main():
    results = {
        "model": "SmallThinker-3B-Preview (Q4_K_M)",
        "fixed_settings": {
            "context": CTX,
            "n_tokens": N_TOKENS,
            "threads": THREADS,
            "note": "32K context FIXED per user instruction. Temperature per-category best from temp sweep.",
        },
        "sweeps": {},
    }

    total_runs = len(PROMPTS) * sum(len(s["values"]) for s in SWEEPS)
    run_num = 0

    for sweep in SWEEPS:
        var = sweep["variable"]
        print(f"\n{'='*60}")
        print(f"  SWEEP: {sweep['label']}")
        print(f"  Values: {sweep['values']}")
        print(f"  {sweep['description']}")
        print(f"{'='*60}")

        sweep_results = {
            "variable": var,
            "label": sweep["label"],
            "values": sweep["values"],
            "description": sweep["description"],
            "runs": [],
        }

        for prompt in PROMPTS:
            for val in sweep["values"]:
                run_num += 1
                # Build kwargs, overriding only the swept variable
                kwargs = {
                    "top_p": DEFAULTS["top_p"],
                    "top_k": DEFAULTS["top_k"],
                    "repeat_penalty": DEFAULTS["repeat_penalty"],
                }
                kwargs[var] = val

                print(f"\n[{run_num}/{total_runs}] {prompt['id']} @ {var}={val} ...")
                sys.stdout.flush()

                try:
                    metrics = run_one(
                        prompt["text"], prompt["temp"],
                        kwargs["top_p"], kwargs["top_k"], kwargs["repeat_penalty"],
                        label=f"{prompt['id']}_{var}_{val}"
                    )
                    print(f"  -> gen={metrics['gen_tps']} t/s, {metrics['wall_time']}s, "
                          f"think={metrics['think_chars']}ch, ans={metrics['ans_chars']}ch "
                          f"({metrics['think_pct']}% think)")

                    sweep_results["runs"].append({
                        "prompt_id": prompt["id"],
                        "prompt_label": prompt["label"],
                        "temp": prompt["temp"],
                        **{var: val},
                        **{k: kwargs[k] for k in DEFAULTS if k != var},  # other defaults
                        **metrics,
                    })
                except subprocess.TimeoutExpired:
                    print(f"  -> TIMEOUT (600s)")
                    sweep_results["runs"].append({
                        "prompt_id": prompt["id"],
                        "prompt_label": prompt["label"],
                        "temp": prompt["temp"],
                        **{var: val},
                        **{k: DEFAULTS[k] for k in DEFAULTS if k != var},
                        "gen_tps": 0, "wall_time": 600, "think_chars": 0,
                        "ans_chars": 0, "think_pct": 0, "truncated": True,
                        "error": "timeout",
                    })
                except Exception as e:
                    print(f"  -> ERROR: {e}")
                    sweep_results["runs"].append({
                        "prompt_id": prompt["id"],
                        "prompt_label": prompt["label"],
                        "temp": prompt["temp"],
                        **{var: val},
                        **{k: DEFAULTS[k] for k in DEFAULTS if k != var},
                        "gen_tps": 0, "wall_time": 0, "think_chars": 0,
                        "ans_chars": 0, "think_pct": 0, "truncated": True,
                        "error": str(e),
                    })

        results["sweeps"][var] = sweep_results

        # Save incremental after each sweep (so progress survives crashes)
        out_path = os.path.expanduser(
            "~/projects/deep-smallthinker-3b/results/variable_sweep_results.json"
        )
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [saved incremental results after {var} sweep]")

    print(f"\n{'='*60}")
    print(f"  ALL SWEEPS COMPLETE — {total_runs} runs total")
    print(f"  Results: results/variable_sweep_results.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()