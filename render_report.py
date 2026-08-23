#!/usr/bin/env python3
"""Merge benchmark metrics + quality scores into a single comparison report."""
import json
import os

RESULTS = os.path.expanduser("~/projects/deep-smallthinker-3b/results/benchmark_results.json")
QUALITY = os.path.expanduser("~/projects/deep-smallthinker-3b/results/quality_scores.json")
OUT = os.path.expanduser("~/projects/deep-smallthinker-3b/results/benchmark_final.json")

runs = json.load(open(RESULTS))["runs"]
q = json.load(open(QUALITY))
scores = q["quality_scores"]

rows = []
for r in runs:
    if "error" in r:
        rows.append({"id": r["id"], "error": r["error"]})
        continue
    sid = r["id"]
    sc = scores.get(sid, {})
    total_chars = r["thinking_chars"] + r["answer_chars"]
    tokens_est = round(total_chars / 4)
    rows.append({
        "id": sid,
        "suite": r["suite"],
        "label": r["label"],
        "gen_tps": r["gen_tps"],
        "prompt_tps": r["prompt_tps"],
        "tokens_est": tokens_est,
        "wall_time_s": r["wall_time_s"],
        "thinking_chars": r["thinking_chars"],
        "answer_chars": r["answer_chars"],
        "think_ratio_pct": round(r["thinking_chars"] / total_chars * 100, 1) if total_chars else 0,
        "quality": sc.get("score"),
        "notes": sc.get("notes", ""),
    })

final = {
    "model": q["model"],
    "settings": q["settings"],
    "average_quality": q["average_quality"],
    "prior_temp_0.3_average": q["prior_temp_0.3_average"],
    "key_finding": q["key_finding"],
    "results": rows,
}
json.dump(final, open(OUT, "w"), indent=2)

# Render the combined table
print(f"\n{'='*100}")
print("DEEP-SMALLTHINKER-3B — FULL BENCHMARK (temp 1.0, 32K ctx, 8192 tok)")
print(f"{'='*100}")
print(f"{'Prompt':<9} {'Suite':<8} {'Gen':>6} {'Ppt':>6} {'Tok*':>6} {'Wall':>6} {'Think%':>7} {'Qual':>5}")
print(f"{'-'*100}")
for r in rows:
    if "error" in r:
        print(f"{r['id']:<9} ERROR: {r['error']}")
        continue
    print(f"{r['id']:<9} {r['suite']:<8} {r['gen_tps']:>6.1f} {r['prompt_tps']:>6.0f} "
          f"{r['tokens_est']:>6} {r['wall_time_s']:>6.0f} {r['think_ratio_pct']:>6.1f}% {r['quality']:>5}")

print(f"{'-'*100}")
gen = [r['gen_tps'] for r in rows if 'gen_tps' in r]
ppt = [r['prompt_tps'] for r in rows if 'prompt_tps' in r]
print(f"{'AVG':<9} {'':<8} {sum(gen)/len(gen):>6.1f} {sum(ppt)/len(ppt):>6.0f} "
      f"{'':>6} {'':>6} {'':>7} {q['average_quality']:>5}")
print(f"{'='*100}")
print(f"*Tok estimated (chars/4). Average quality: {q['average_quality']}/10 "
      f"(prior temp-0.3 run: {q['prior_temp_0.3_average']}/10)")
print(f"\nKey finding: {q['key_finding']}")
print(f"\nSaved: {OUT}")
