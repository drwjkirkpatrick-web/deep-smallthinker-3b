#!/usr/bin/env python3
"""
Temperature sweep for SmallThinker-3B Q4_K_M on the open-ended prompts that
collapsed into meta-reasoning loops at temp 1.0 (iambic, creative, basic, prose).

ONLY the temperature varies. Every other parameter is held constant at the
deep-thinking values from the main benchmark:
  - context 32768
  - n_tokens 8192
  - top_p 0.95, top_k 40, repeat_penalty 1.1
  - threads 6, ngl 99, fa on, --jinja, --no-conversation, -st

Temperatures swept: 0.1, 0.3, 0.5, 0.7, 0.9  (1.0 already measured in main bench)

For each run we capture the same metrics as the main benchmark (gen_tps,
prompt_tps, thinking/answer chars, think ratio) plus the full output text so
quality can be scored 1-10 afterward.
"""

import json
import os
import re
import subprocess
import time

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
MODEL = os.path.expanduser("~/models/smallthinker-3b-q4_k_m.gguf")
OUT_DIR = os.path.expanduser("~/projects/deep-smallthinker-3b/results")
RESULTS_FILE = os.path.join(OUT_DIR, "temp_sweep_results.json")

# Fixed settings — ONLY temp changes between runs
FIXED = {
    "ctx": 32768,
    "n_tokens": 8192,
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "threads": 6,
}

TEMPS = [0.1, 0.3, 0.5, 0.7, 0.9]

# The four open-ended prompts (identical text to main benchmark)
PROMPTS = [
    ("iambic", "Iambic Pentameter",
     "Write a poem about the changing of seasons from autumn to winter, strictly in iambic pentameter (10 syllables per line, alternating stress). Write exactly 8 lines with an ABAB CDCD rhyme scheme."),
    ("creative", "Creative Writing",
     "Write a short scene (200-300 words) set in a lighthouse during a storm. The lighthouse keeper is an old woman who has been alone for thirty years. A stranger arrives at the door, drenched and shivering. Show, don't tell — use sensory details and subtext."),
    ("basic", "TRS-80 BASIC Retro Game",
     "Write a TRS-80 Model III BASIC program for a simple text adventure game. "
     "The player explores 3 rooms (Entrance Hall, Library, Treasure Room) looking for a golden key. "
     "Use PRINT for room descriptions, INPUT for player commands (GO NORTH, GO SOUTH, TAKE KEY, EXAMINE), "
     "string variables for room descriptions, and GOTO for navigation. "
     "Include inventory tracking and a win condition when the player takes the key "
     "and reaches the Treasure Room. Number lines starting at 10 with increments of 10."),
    ("prose", "Clinical Prose",
     "Explain the pathophysiology of Hashimoto's thyroiditis in detail. Cover the autoimmune mechanism, the role of anti-TPO antibodies, the progression from euthyroid to hypothyroid, and the typical lab findings at each stage. Write for a medical student audience."),
]


def strip_banner(text):
    text = re.sub(r'Loading model\.\.\..*?(?=\n\n|\n[▄█])', '', text, flags=re.DOTALL)
    text = re.sub(r'▄▄ ▄▄.*?(?=build|available|>)', '', text, flags=re.DOTALL)
    text = re.sub(r'^build\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^model\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^ftype\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^modalities\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^available commands:.*?(?=\n\n|\n>|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'^system\s*:.*?(?=\n\n|\nuser|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)
    lines = text.split('\n')
    prompt_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('> ') or s == '>':
            prompt_idx = i
    if prompt_idx >= 0:
        lines = lines[prompt_idx + 1:]
    text = '\n'.join(lines)
    for marker in ('<|im_start|>', '<|im_end|>', '<|channel|>',
                   '<|start_header_id|>', '<|end_header_id|>', '<|endoftext|>'):
        text = text.replace(marker, '')
    return text.strip()


def split_thinking_answer(text):
    for marker in ('**Final Answer**', 'Final Answer', '\\boxed{', '**Answer**',
                   'The answer is', '```\n'):
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx].strip(), text[idx:].strip()
    return text.strip(), ""


def run(pid, label, prompt_text, temp):
    cmd = [
        LLAMA_CLI, "-m", MODEL, "-p", prompt_text,
        "-n", str(FIXED["n_tokens"]),
        "-c", str(FIXED["ctx"]),
        "--temp", str(temp),
        "--top-p", str(FIXED["top_p"]),
        "--top-k", str(FIXED["top_k"]),
        "--repeat-penalty", str(FIXED["repeat_penalty"]),
        "-t", str(FIXED["threads"]),
        "-ngl", "99", "-fa", "on",
        "--jinja", "--no-conversation", "--no-display-prompt", "-st",
    ]
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        wall = time.time() - start
    except subprocess.TimeoutExpired:
        return {"id": pid, "temp": temp, "error": "TIMEOUT (900s)"}
    except Exception as e:
        return {"id": pid, "temp": temp, "error": str(e)}

    raw = result.stdout + result.stderr
    gen_tps = prompt_tps = 0.0
    m = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', raw)
    if m:
        prompt_tps = float(m.group(1))
        gen_tps = float(m.group(2))

    clean = strip_banner(raw)
    clean = re.sub(r'\[ Prompt:.*Generation:.*\].*', '', clean, flags=re.DOTALL).strip()
    clean = re.sub(r'Exiting\.\.\.', '', clean).strip()
    thinking, answer = split_thinking_answer(clean)
    total_chars = len(thinking) + len(answer)

    return {
        "id": pid,
        "label": label,
        "temp": temp,
        "gen_tps": round(gen_tps, 2),
        "prompt_tps": round(prompt_tps, 2),
        "wall_time_s": round(wall, 1),
        "thinking_chars": len(thinking),
        "answer_chars": len(answer),
        "think_ratio_pct": round(len(thinking) / total_chars * 100, 1) if total_chars else 0,
        "output_text": clean,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {
        "model": "SmallThinker-3B-Preview (Q4_K_M)",
        "fixed_settings": FIXED,
        "note": "Temperature sweep on open-ended prompts. ONLY temp varies.",
        "temps": TEMPS,
        "runs": [],
    }

    total = len(PROMPTS) * len(TEMPS)
    i = 0
    for pid, label, prompt_text in PROMPTS:
        for temp in TEMPS:
            i += 1
            print(f"[{i}/{total}] {pid} @ temp={temp} ...", flush=True)
            r = run(pid, label, prompt_text, temp)
            data["runs"].append(r)
            if "error" in r:
                print(f"  -> ERROR: {r['error']}")
            else:
                print(f"  -> gen={r['gen_tps']} t/s, {r['wall_time_s']}s, "
                      f"think={r['thinking_chars']}ch, ans={r['answer_chars']}ch "
                      f"({r['think_ratio_pct']}% think)")
            with open(RESULTS_FILE, "w") as f:
                json.dump(data, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
