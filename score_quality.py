#!/usr/bin/env python3
"""
Post-process deep-smallthinker-3b benchmark results:
  1. Estimate token counts (n_eval is not printed by this llama.cpp build).
  2. Compute thinking/answer ratios and other additional metrics.
  3. Print a clean summary table.

Token estimate: English text averages ~4 chars/token, so
  tokens_est = round(chars / 4). Clearly labeled as an estimate.
"""

import json
import os

RESULTS_FILE = os.path.expanduser("~/projects/deep-smallthinker-3b/results/benchmark_results.json")
OUT_FILE = os.path.expanduser("~/projects/deep-smallthinker-3b/results/benchmark_summary.json")


def load():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def main():
    data = load()
    runs = data["runs"]

    summary = []
    for r in runs:
        if "error" in r:
            summary.append({"id": r["id"], "error": r["error"]})
            continue
        total_chars = r["thinking_chars"] + r["answer_chars"]
        tokens_est = round(total_chars / 4)
        think_ratio = round(r["thinking_chars"] / total_chars * 100, 1) if total_chars else 0
        summary.append({
            "id": r["id"],
            "suite": r["suite"],
            "label": r["label"],
            "gen_tps": r["gen_tps"],
            "prompt_tps": r["prompt_tps"],
            "tokens_est": tokens_est,
            "wall_time_s": r["wall_time_s"],
            "thinking_chars": r["thinking_chars"],
            "answer_chars": r["answer_chars"],
            "think_ratio_pct": think_ratio,
            "hit_max": r["hit_max"],
            "quality": None,  # filled in after manual scoring
            "notes": "",
        })

    # Print speed + metrics table
    print(f"\n{'='*95}")
    print("DEEP-SMALLTHINKER-3B BENCHMARK — SPEED & METRICS (temp 1.0, 32K ctx, 8192 tok)")
    print(f"{'='*95}")
    print(f"{'Prompt':<10} {'Suite':<8} {'Gen t/s':>8} {'Ppt t/s':>8} {'Tok*':>6} {'Wall s':>7} {'Think%':>7} {'Ans ch':>7}")
    print(f"{'-'*95}")
    gen_speeds = []
    prompt_speeds = []
    for s in summary:
        if "error" in s:
            print(f"{s['id']:<10} {'—':<8} {'ERROR':>8} {s['error']}")
            continue
        print(f"{s['id']:<10} {s['suite']:<8} {s['gen_tps']:>8.1f} {s['prompt_tps']:>8.1f} "
              f"{s['tokens_est']:>6} {s['wall_time_s']:>7.1f} {s['think_ratio_pct']:>6.1f}% {s['answer_chars']:>7}")
        gen_speeds.append(s["gen_tps"])
        prompt_speeds.append(s["prompt_tps"])

    if gen_speeds:
        print(f"{'-'*95}")
        print(f"{'AVG':<10} {'':<8} {sum(gen_speeds)/len(gen_speeds):>8.1f} "
              f"{sum(prompt_speeds)/len(prompt_speeds):>8.1f}")
    print(f"{'='*95}")
    print("*Tok = estimated from chars/4 (this llama.cpp build does not print n_eval)")

    with open(OUT_FILE, "w") as f:
        json.dump({"model": data["model"], "settings": data["settings"],
                   "summary": summary}, f, indent=2)
    print(f"\nSummary saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
