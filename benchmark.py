#!/usr/bin/env python3
"""
Deep-thinking benchmark for SmallThinker-3B Q4_K_M on 8GB Jetson.

Runs the SAME 10 prompts used across our prior LLM benchmark projects:
  - General 5-prompt suite (multiprompt_bench.py): code, iambic pentameter,
    clinical prose, creative writing, math proof
  - Coding 5-prompt suite (bench_coding.py): HTML/CSS, Python, C, TRS-80 BASIC, Julia

Difference from the prior runs: this uses the DEEP-THINKING settings
(temp 1.0, 32K context, 8192 excess thinking tokens) that are the whole point
of the deep-smallthinker-3b project, instead of the temp 0.3 used before.

Captures per prompt:
  - gen_tps        (generation tokens/sec)
  - prompt_tps     (prompt-eval tokens/sec)
  - n_eval         (actual tokens generated)
  - wall_time_s    (end-to-end wall clock)
  - hit_max        (whether it hit the -n token ceiling = truncated thinking)
  - thinking_chars (reasoning length before the final answer)
  - answer_chars   (final-answer length)
  - output_text    (full stripped output for quality scoring)

Quality (1-10) is scored separately after the runs complete.
"""

import json
import os
import re
import subprocess
import sys
import time

LLAMA_CLI = os.path.expanduser("~/llama.cpp/build/bin/llama-cli")
MODEL = os.path.expanduser("~/models/smallthinker-3b-q4_k_m.gguf")
OUT_DIR = os.path.expanduser("~/projects/deep-smallthinker-3b/results")
RESULTS_FILE = os.path.join(OUT_DIR, "benchmark_results.json")

# Deep-thinking settings (match think.sh `max` mode, bounded at 8192 for runtime)
SETTINGS = {
    "n_tokens": 8192,      # excess thinking tokens (max mode default is 16384)
    "ctx": 32768,          # native 32K context
    "temp": 1.0,           # high temp: tokens are free, favor exploration
    "top_p": 0.95,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "threads": 6,
}

# --- Prompt suites (identical text to prior benchmark scripts) ---
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
    # Drop the ASCII logo block and header lines
    text = re.sub(r'Loading model\.\.\..*?(?=\n\n|\n[▄█])', '', text, flags=re.DOTALL)
    text = re.sub(r'▄▄ ▄▄.*?(?=build|available|>)', '', text, flags=re.DOTALL)
    text = re.sub(r'^build\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^model\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^ftype\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^modalities\s+:.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^available commands:.*?(?=\n\n|\n>|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'^system\s*:.*?(?=\n\n|\nuser|\Z)', '', text, flags=re.DOTALL | re.MULTILINE)
    # Cut at the prompt echo (first '> ' line)
    lines = text.split('\n')
    prompt_idx = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith('> ') or s == '>':
            prompt_idx = i
    if prompt_idx >= 0:
        lines = lines[prompt_idx + 1:]
    text = '\n'.join(lines)
    # Strip token markers
    for marker in ('<|im_start|>', '<|im_end|>', '<|channel|>',
                   '<|start_header_id|>', '<|end_header_id|>', '<|endoftext|>'):
        text = text.replace(marker, '')
    return text.strip()


def split_thinking_answer(text):
    """Split output into thinking (before the final answer) and final answer."""
    # SmallThinker signals the final answer with these markers
    for marker in ('**Final Answer**', 'Final Answer', '\\boxed{', '**Answer**',
                   'The answer is', '```\n'):
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx].strip(), text[idx:].strip()
    return text.strip(), ""


def run_prompt(suite, pid, label, prompt_text):
    cmd = [
        LLAMA_CLI, "-m", MODEL, "-p", prompt_text,
        "-n", str(SETTINGS["n_tokens"]),
        "-c", str(SETTINGS["ctx"]),
        "--temp", str(SETTINGS["temp"]),
        "--top-p", str(SETTINGS["top_p"]),
        "--top-k", str(SETTINGS["top_k"]),
        "--repeat-penalty", str(SETTINGS["repeat_penalty"]),
        "-t", str(SETTINGS["threads"]),
        "-ngl", "99", "-fa", "on",
        "--jinja", "--no-conversation", "--no-display-prompt", "-st",
    ]
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        wall = time.time() - start
    except subprocess.TimeoutExpired:
        return {"suite": suite, "id": pid, "label": label, "error": "TIMEOUT (900s)"}
    except Exception as e:
        return {"suite": suite, "id": pid, "label": label, "error": str(e)}

    raw = result.stdout + result.stderr

    # Speed metrics
    gen_tps = prompt_tps = 0.0
    m = re.search(r'Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s', raw)
    if m:
        prompt_tps = float(m.group(1))
        gen_tps = float(m.group(2))

    # Token count (n_eval in stderr perf printout)
    n_eval = 0
    m = re.search(r'n_eval\s*=\s*(\d+)', raw)
    if m:
        n_eval = int(m.group(1))

    # Strip banner to get the clean answer
    clean = strip_banner(raw)
    # Remove the trailing timing line + "Exiting..." from clean output
    clean = re.sub(r'\[ Prompt:.*Generation:.*\].*', '', clean, flags=re.DOTALL).strip()
    clean = re.sub(r'Exiting\.\.\.', '', clean).strip()

    thinking, answer = split_thinking_answer(clean)
    hit_max = n_eval >= SETTINGS["n_tokens"]

    return {
        "suite": suite,
        "id": pid,
        "label": label,
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

    data = {
        "model": "SmallThinker-3B-Preview (Q4_K_M)",
        "gguf": MODEL,
        "settings": SETTINGS,
        "note": "Deep-thinking settings (temp 1.0, 32K ctx, 8192 tokens) — NOT the temp 0.3 used in prior benchmarks",
        "runs": [],
    }

    total = len(ALL_PROMPTS)
    for i, (suite, (pid, label, prompt_text)) in enumerate(ALL_PROMPTS, 1):
        print(f"[{i}/{total}] {suite}/{pid} ({label}) ...", flush=True)
        r = run_prompt(suite, pid, label, prompt_text)
        data["runs"].append(r)
        if "error" in r:
            print(f"  -> ERROR: {r['error']}")
        else:
            print(f"  -> gen={r['gen_tps']} t/s, prompt={r['prompt_tps']} t/s, "
                  f"{r['n_eval']} tok, {r['wall_time_s']}s, "
                  f"think={r['thinking_chars']}ch, ans={r['answer_chars']}ch"
                  + ("  [HIT MAX]" if r["hit_max"] else ""))
        # Save incrementally
        with open(RESULTS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
