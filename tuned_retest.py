#!/usr/bin/env python3
"""
Re-benchmark the original 10 prompts with TUNED settings.

For each prompt:
  1. Classify it using auto_temp.py's classifier
  2. Apply the empirically-best settings from all 65 experiments:
     - temperature (from temp sweep)
     - top_p, top_k, repeat_penalty (from variable sweep)
  3. Keep 32K context fixed (per user instruction)
  4. Capture the same metrics as the original benchmark

Then we compare old (temp 1.0, default sampling) vs new (tuned) quality.
"""
import json
import os
import re
import subprocess
import sys
import time

# Import the classifier from auto_temp.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_temp import classify

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
MODEL = os.path.expanduser("~/models/smallthinker-3b-q4_k_m.gguf")
OUT_DIR = os.path.expanduser("~/projects/deep-smallthinker-3b/results")
RESULTS_FILE = os.path.join(OUT_DIR, "tuned_retest_results.json")

# ── Tuned settings per category (from 65 experiments) ──────────────────
# Format: (temp, top_p, top_k, repeat_penalty, tokens)
# Context is ALWAYS 32768 (user instruction: "always keep the 32k context")
TUNED = {
    "reasoning": {"temp": 1.0,  "top_p": 0.80, "top_k": 80, "repeat_penalty": 1.10, "tokens": 8192},
    "fiction":   {"temp": 0.9,  "top_p": 0.90, "top_k": 80, "repeat_penalty": 1.20, "tokens": 8192},
    "poetry":    {"temp": 0.3,  "top_p": 0.90, "top_k": 80, "repeat_penalty": 1.20, "tokens": 8192},
    "prose":     {"temp": 0.5,  "top_p": 0.90, "top_k": 80, "repeat_penalty": 1.20, "tokens": 8192},
    "code":      {"temp": 0.7,  "top_p": 1.00, "top_k": 40, "repeat_penalty": 1.20, "tokens": 8192},
}

CONTEXT = 32768   # ALWAYS 32K per user instruction
THREADS = 6

# ── The same 10 prompts from benchmark.py ──────────────────────────────
GENERAL_PROMPTS = [
    ("code", "Code Generation",
     "Write a Python function called `merge_sort` that takes a list of integers and returns a new list sorted in ascending order. Include type hints, a docstring, and handle edge cases (empty list, single element). Then show an example usage."),
    ("iambic", "Iambic Pentameter",
     "Write a poem about the changing of seasons from autumn to winter, strictly in iambic pentameter (10 syllables per line, alternating stress). Write exactly 8 lines with an ABAB CDCD rhyme scheme."),
    ("prose", "Clinical Prose",
     "Explain the pathophysiology of Hashimoto's thyroiditis in detail. Cover the autoimmune mechanism, the role of anti-TPO antibodies, the progression from euthyroid to hypothyroid, and the typical lab findings at each stage. Write for a medical student audience."),
    ("creative", "Creative Writing",
     "Write a short scene (200-300 words) set in a lighthouse during a storm. The lighthouse keeper is an old woman who has been alone for thirty years. A stranger arrives at the door, drenched and shivering. Show, don't tell — use sensory details and subtext."),
    ("math", "Mathematical Proof",
     "Prove that the square root of 2 is irrational. Use a proof by contradiction. State each step clearly with justification. Begin with: 'Theorem: sqrt(2) is irrational.'"),
]

CODING_PROMPTS = [
    ("html", "HTML/CSS Web Page",
     "Create a complete HTML page with embedded CSS for a personal portfolio website. "
     "Include a header with navigation, a hero section with name and tagline, "
     "an about section, a projects grid with 3 sample project cards, "
     "and a footer with contact links. Use modern CSS with flexbox, "
     "a color scheme of dark blue and gold, and responsive design with a media query for mobile."),
    ("python", "Python Data Processing",
     "Write a Python class called DataProcessor that loads a CSV file of employee records "
     "(columns: name, department, salary, hire_date), filters employees hired after 2020, "
     "calculates average salary by department, finds the top 3 highest paid employees, "
     "and exports results to a JSON file. Include type hints, docstrings, error handling, "
     "and use the csv and json standard library modules. Show example usage."),
    ("c", "C System Programming",
     "Write a C program that implements a simple thread-safe queue using a linked list. "
     "Include functions: queue_init, queue_push, queue_pop, queue_size, and queue_destroy. "
     "Use pthread mutex for thread safety. Include a main function that demonstrates "
     "creating the queue, pushing 5 integers, and popping them all. "
     "Add proper error handling and memory management (free on destroy)."),
    ("basic", "TRS-80 BASIC Retro Game",
     "Write a TRS-80 Model III BASIC program for a simple text adventure game. "
     "The player explores 3 rooms (Entrance Hall, Library, Treasure Room) looking for a golden key. "
     "Use PRINT for room descriptions, INPUT for player commands (GO NORTH, GO SOUTH, TAKE KEY, EXAMINE), "
     "string variables for room descriptions, and GOTO for navigation. "
     "Include inventory tracking and a win condition when the player takes the key "
     "and reaches the Treasure Room. Number lines starting at 10 with increments of 10."),
    ("julia", "Julia Numerical Computing",
     "Write a Julia program that implements the Newton-Raphson method for finding roots "
     "of a function. Create a function newton_raphson(f, df, x0, tol, max_iter) that takes "
     "a function, its derivative, initial guess, tolerance, and max iterations. "
     "Use it to find the root of x^3 - 2x - 5 (starting at x0=2.0). "
     "Print each iteration showing the step number, current x, f(x), and the error. "
     "Include type annotations and a docstring. Show the final result."),
]

ALL_PROMPTS = [("general", p) for p in GENERAL_PROMPTS] + [("coding", p) for p in CODING_PROMPTS]


def strip_banner(text):
    """Remove llama.cpp banner + UI chrome, keep only the generated response."""
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
                   '<|start_header_id|>', '<|end_header_id|>', ''):
        text = text.replace(marker, '')
    return text.strip()


def split_thinking_answer(text):
    """Split output into thinking and final answer."""
    for marker in ('**Final Answer**', 'Final Answer', '\\boxed{', '**Answer**',
                   'The answer is', '```\n'):
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx].strip(), text[idx:].strip()
    # If no marker found, check for </think> tag (SmallThinker's native format)
    idx = text.find('</think>')
    if idx >= 0:
        return text[:idx].strip(), text[idx + len('</think>'):].strip()
    return text.strip(), ""


def run_prompt(suite, pid, label, prompt_text):
    # Classify the prompt
    temp, style, ctx, tokens = classify(prompt_text)
    tuned = TUNED[style]

    # Build the command with TUNED settings
    cmd = [
        LLAMA_CLI, "-m", MODEL, "-p", prompt_text,
        "-n", str(tuned["tokens"]),
        "-c", str(CONTEXT),             # ALWAYS 32K
        "--temp", str(tuned["temp"]),
        "--top-p", str(tuned["top_p"]),
        "--top-k", str(tuned["top_k"]),
        "--repeat-penalty", str(tuned["repeat_penalty"]),
        "-t", str(THREADS),
        "-ngl", "99", "-fa", "on",
        "--jinja", "--no-conversation", "--no-display-prompt", "-st",
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        wall = time.time() - start
    except subprocess.TimeoutExpired:
        return {"suite": suite, "id": pid, "label": label, "error": "TIMEOUT (900s)",
                "style": style, "settings": tuned}
    except Exception as e:
        return {"suite": suite, "id": pid, "label": label, "error": str(e),
                "style": style, "settings": tuned}

    raw = result.stdout + result.stderr

    # Speed metrics
    gen_tps = prompt_tps = 0.0
    m = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', raw)
    if m:
        prompt_tps = float(m.group(1))
        gen_tps = float(m.group(2))

    n_eval = 0
    m = re.search(r'n_eval\s*=\s*(\d+)', raw)
    if m:
        n_eval = int(m.group(1))

    clean = strip_banner(raw)
    clean = re.sub(r'\[ Prompt:.*Generation:.*\].*', '', clean, flags=re.DOTALL).strip()
    clean = re.sub(r'Exiting\.\.\.', '', clean).strip()

    thinking, answer = split_thinking_answer(clean)
    hit_max = n_eval >= tuned["tokens"]

    return {
        "suite": suite,
        "id": pid,
        "label": label,
        "style": style,
        "settings": tuned,
        "gen_tps": round(gen_tps, 2),
        "prompt_tps": round(prompt_tps, 2),
        "n_eval": n_eval,
        "wall_time_s": round(wall, 1),
        "hit_max": hit_max,
        "thinking_chars": len(thinking),
        "answer_chars": len(answer),
        "output_text": clean,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # First, show the classification for each prompt
    print("=" * 60)
    print("  TUNED RETEST — 10 prompts with best settings per category")
    print("  Context: 32K (fixed) | Sampling: tuned per category")
    print("=" * 60)

    classifications = []
    for suite, (pid, label, prompt_text) in ALL_PROMPTS:
        temp, style, ctx, tokens = classify(prompt_text)
        tuned = TUNED[style]
        print(f"  {pid:12s} → {style:10s} | temp={tuned['temp']} top_p={tuned['top_p']} "
              f"top_k={tuned['top_k']} rep={tuned['repeat_penalty']}")
        classifications.append((pid, style, tuned))

    print("=" * 60)
    print()

    data = {
        "model": "SmallThinker-3B-Preview (Q4_K_M)",
        "gguf": MODEL,
        "context": CONTEXT,
        "note": "Retest of original 10 prompts with TUNED settings per category. "
                "Context always 32K. Sampling params from 65-experiment tuning.",
        "classifications": {pid: {"style": style, "settings": tuned}
                            for pid, style, tuned in classifications},
        "runs": [],
    }

    total = len(ALL_PROMPTS)
    for i, (suite, (pid, label, prompt_text)) in enumerate(ALL_PROMPTS, 1):
        temp, style, _, _ = classify(prompt_text)
        tuned = TUNED[style]
        print(f"[{i}/{total}] {suite}/{pid} ({label}) → {style} "
              f"temp={tuned['temp']} top_p={tuned['top_p']} "
              f"top_k={tuned['top_k']} rep={tuned['repeat_penalty']} ...", flush=True)
        r = run_prompt(suite, pid, label, prompt_text)
        data["runs"].append(r)
        if "error" in r:
            print(f"  -> ERROR: {r['error']}")
        else:
            print(f"  -> gen={r['gen_tps']} t/s, {r['n_eval']} tok, {r['wall_time_s']}s, "
                  f"think={r['thinking_chars']}ch, ans={r['answer_chars']}ch"
                  + ("  [HIT MAX]" if r["hit_max"] else ""))
        # Save incrementally
        with open(RESULTS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()