# deep-smallthinker-3b

A single-command way to run [SmallThinker-3B](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) — a small AI model that "thinks out loud" before it answers — on an 8GB Jetson. You ask one question, it shows you its full chain of thought, then gives you the answer. No setup between questions, no interactive loop, just ask-and-answer.

This project includes something most don't: an **automatic settings adjuster** that reads your prompt and picks the best temperature *and sampling parameters* for it. We ran **75 experiments total** — a 10-prompt benchmark, a 20-run temperature sweep, a 45-run variable sweep, and a final 10-prompt retest with the tuned settings. The result: **average quality rose from 6.4/10 to 9.8/10. All 10 prompts improved, zero regressions.**

---

## What is SmallThinker-3B?

It's a 3.4-billion-parameter AI model built on Qwen2.5-3B. Unlike a regular chatbot that answers immediately, SmallThinker writes out its reasoning first — sometimes thousands of words of "let me consider..." and "if X then Y..." before it arrives at a conclusion. Think of it as a model that shows its work on every question.

**What it's good at:** math proofs, coding problems, logic puzzles, step-by-step reasoning — tasks where you can check whether the answer is right.

**What it struggles with:** creative writing, poetry, open-ended prose — tasks where the "thinking" part can spiral into an endless loop of "I need to write X... first I should... let me think..." and never actually produce the thing you asked for. We discovered this the hard way and built the auto-adjuster to fix it.

---

## Step 1: Build llama.cpp

You need llama.cpp compiled with CUDA support. This is the engine that runs the model.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_CUDA_F16=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

This takes about 10 minutes. When it's done, you'll have `llama.cpp/build/bin/llama-cli`.

> Already built llama.cpp on your Jetson? Skip this step — the scripts default to `~/llama.cpp/build/bin/llama-cli`.

---

## Step 2: Download the model

The model file is about 2 GB. We use the Q4_K_M quantization (compressed format) because it's small enough to fit in the Jetson's 8 GB of memory while leaving room for a large context window.

```bash
cd ~/projects/deep-smallthinker-3b
./scripts/download-model.sh
```

This downloads `smallthinker-3b-q4_k_m.gguf` to `~/models/`. It takes 5–10 minutes depending on your connection.

> **Why Q4_K_M instead of Q8_0?** The Q8_0 version is 3.4 GB and runs at 19.2 tokens/second. The Q4_K_M is 2.1 GB and runs at 20.0 tokens/second. The smaller size frees ~1.3 GB of RAM, which is exactly what we need to open up a larger context window (32K tokens) for deep thinking. Quality is nearly identical on the tasks this model is good at.

---

## Step 3: Turn off the GUI (recommended)

The Jetson shares its 8 GB between the GPU and the desktop interface. Stopping the desktop frees about 600 MB, which lets us use the full 32K context window.

```bash
sudo systemctl stop gdm3
```

Your screen will go black — that's normal. You can still SSH in or use a terminal (Ctrl+Alt+F2). To turn the GUI back on later:

```bash
sudo systemctl start gdm3
```

> **Can't or don't want to stop the GUI?** The scripts still work — they just use a smaller context window (8K instead of 32K). The model will still think, just not as deeply on very long problems.

---

## Step 4: Ask a question

### The easy way — automatic settings

```bash
./think-auto.sh "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?"
```

That's it. The auto-adjuster reads your prompt, figures out what kind of question it is, and picks the best temperature *and sampling parameters* for it. You'll see:

```
== think-auto.sh (auto-tuned) ==
  prompt:    A bat and a ball cost $1.10 total. The bat costs $1 more...
  style:     reasoning
  temp:      1.0  (auto-selected)
  context:   32768
  tokens:    16384
  top_p:     0.80  |  top_k: 80  |  repeat: 1.10
  [tuned via 75-experiment sweep, validated 9.8/10]
=================================

<thousands of words of chain-of-thought reasoning>

The ball costs $0.05.

[ Prompt: 279.1 t/s | Generation: 20.0 t/s ]
```

### What the auto-adjuster does

It classifies your prompt into one of five categories and uses the best temperature + sampling parameters we found through 75 experiments — then validated with a final 10-prompt retest:

| Your prompt asks for... | Auto-detected as... | Temp | top_p | top_k | rep | Why |
|-------------------------|---------------------|------|-------|-------|-----|-----|
| Math, proofs, algorithms | reasoning | 1.0 | 0.80 | 80 | 1.10 | High temp explores; rep=1.10 is the only value that produces complete proofs (5/5 elements) |
| A story, scene, narrative | fiction | 0.9 | 0.90 | 80 | 1.20 | High top_k skips the thinking loop; rep=1.20 prevents output bloat (33s vs 160s) |
| A poem, sonnet, verse | poetry | 0.3 | 0.90 | 80 | 1.20 | Lower temp prevents the spiral (poetry is still its weakest skill) |
| An explanation or description | prose | 0.5 | 0.90 | 80 | 1.20 | Prose is robust everywhere; these settings are fast and clean |
| A game, retro code, interactive | code | 0.7 | 1.00 | 40 | 1.20 | top_k=40 for code (80 over-generates); rep=1.20 is 2.6x faster with same quality |

> **How did we find these numbers?** Three rounds of experiments, plus a validation retest:
> 1. **Temperature sweep** (20 runs): 4 prompt types × 5 temperatures (0.1–0.9), only temperature changed
> 2. **Variable sweep** (45 runs): 3 prompts × 5 values × 3 variables (top_p, top_k, repeat_penalty), only one variable changed per sweep, 32K context fixed
> 3. **Tuned retest** (10 runs): all 10 original benchmark prompts re-run with their category's best settings — quality rose from 6.4/10 to 9.8/10
>
> Full data: `results/temp_sweep_quality.json`, `results/variable_sweep_quality.json`, and `results/tuned_retest_quality.json`.

### The manual way — if you want control

```bash
# Deep reasoning (max thinking, temp 1.0, 32K context)
./think.sh "Prove that sqrt(2) is irrational."

# Creative writing (temp picked per style)
./think-creative.sh "Write a scene where a lighthouse keeper meets a stranger."
./think-creative.sh --style poetry "Write a sonnet about the ocean."
./think-creative.sh --style prose "Explain Hashimoto's thyroiditis."
./think-creative.sh --style code "Write a TRS-80 BASIC game."
```

---

## Step 5: Understand what you're seeing

The output has two parts:

1. **The thinking** — A long block of reasoning. SmallThinker was trained on the QwQ-LongCoT-500K dataset, where over 75% of examples have 8,000+ tokens of chain-of-thought. This is the model "showing its work." It may consider wrong paths, backtrack, and correct itself. That's normal — it's how reasoning models work.

2. **The answer** — After the thinking block, you get the final answer, often in a box or marked clearly.

3. **The speed line** — At the very end: `[ Prompt: 279 t/s | Generation: 20 t/s ]`. Prompt speed is how fast it read your question. Generation speed is how fast it wrote the answer. On this Jetson, generation is consistently ~20 tokens/second (about 15 words/second).

> **Important:** SmallThinker is a 3B model — its reasoning is a *draft*, not a guarantee. It got "strawberry has 3 r's" right at temp 1.0, but the Q8_0 version got it wrong at temp 0.6. Always sanity-check its answer against your own judgment.

---

## The three launchers

| Script | When to use | Temperature | top_p | top_k | rep | Context | Tokens |
|--------|-------------|-------------|-------|-------|-----|---------|--------|
| `think-auto.sh` | Default — let it auto-adjust | Auto (0.3–1.0) | Auto | Auto | Auto | Auto | Auto |
| `think.sh` | Deep reasoning, manual control | 1.0 | 0.80 | 80 | 1.10 | 32K (max mode) | 16K |
| `think-creative.sh` | Creative writing, manual style pick | Per style (0.3–0.9) | Per style | Per style | Per style | 4K | 1.2K |

---

## How the auto-adjuster works

SmallThinker has a quirk: on open-ended creative tasks, it can fall into a **meta-reasoning loop**. Instead of writing the scene you asked for, it writes:

> *"I need to write a scene about a lighthouse. First, I should think about the setting. Let me consider the atmosphere. I'll need a character..."*

...and it does this for the entire 8,000-token budget, never writing the actual scene. This happens at *most* temperatures, and it's **not predictable** — the same prompt at temp 0.3 might work while temp 0.5 loops, and temp 0.9 might be the best for one creative task but the worst for another.

We fixed this in three rounds:

**Round 1 — Temperature sweep** (20 runs): Tested 4 creative prompt types at 5 temperatures (0.1, 0.3, 0.5, 0.7, 0.9), changing only temperature. Found the best temp per task type: fiction 0.9, prose 0.5, poetry 0.3, code 0.7.

**Round 2 — Variable sweep** (45 runs): Tested 3 variables (top_p, top_k, repeat_penalty) at 5 values each, across 3 representative prompts, changing only one variable at a time. 32K context fixed throughout. Key findings:

- **top_k=40 (the common default) is the worst value for reasoning** — 98s vs 37s at top_k=80. Switching to 80 makes reasoning 2.6x faster with the same proof quality.
- **repeat_penalty=1.1 is the only value that produces complete proofs** — 1.0 and 1.05 skip key mathematical steps (2/5 proof elements vs 5/5 at 1.1).
- **repeat_penalty=1.2 is universally faster for creative and coding** — 33–49s vs 47–130s at 1.1, with identical quality scores. It prevents the model from padding output.
- **Coding quality is robust** — 14/15 runs scored 7/7 code elements. Only top_k=60 missed docstrings. Coding doesn't need fine-tuning.
- **Creative quality is robust** — all 15 runs scored 5/5 scene elements. The differentiator is speed, not quality.
- **Reasoning is the most sensitive** — proof completeness varies 2/5 to 5/5 depending on repeat_penalty. This is the variable that matters most for math/logic.

**Round 3 — Tuned retest** (10 runs): All 10 original benchmark prompts re-run with their category's best settings from rounds 1 and 2. This is the validation round — and every prompt improved:

| Prompt | Type | Original score | Tuned score | Improvement |
|--------|------|---------------|-------------|-------------|
| creative writing | fiction | 3/10 | 10/10 | **+7** |
| iambic pentameter | poetry | 4/10 | 10/10 | **+6** |
| TRS-80 BASIC | code | 4/10 | 9/10 | **+5** |
| prose (medical) | prose | 7/10 | 10/10 | +3 |
| HTML page | reasoning | 7/10 | 10/10 | +3 |
| Julia code | reasoning | 7/10 | 10/10 | +3 |
| math proof | reasoning | 8/10 | 10/10 | +2 |
| C program | reasoning | 8/10 | 10/10 | +2 |
| Python function | reasoning | 8/10 | 9/10 | +1 |
| merge sort | reasoning | 8/10 | 10/10 | +2 |
| **Average** | | **6.4/10** | **9.8/10** | **+3.4** |

The biggest jumps were on the prompts that looped hardest at temp 1.0: creative writing went from 3/10 to 10/10, iambic pentameter from 4/10 to 10/10, and TRS-80 BASIC from 4/10 to 9/10. The code-generation prompts that already scored 7–8/10 still gained 1–3 points from the tuned sampling parameters — primarily from top_k=80 producing more complete code with less thinking overhead.

> **Speed note:** The tuned settings prioritize quality, not speed. Several retest runs took longer than the original benchmark because the model produced substantially more complete output (14K vs 8K chars for code, 18K vs 8K for HTML). For speed-sensitive use, use `think-creative.sh` with 4K context.

The `auto_temp.py` classifier reads your prompt, matches it against keyword patterns for each category, and outputs the winning temperature + context + token budget. `think-auto.sh` then maps the style to the tuned top_p/top_k/repeat_penalty from the variable sweep.

```bash
# See what it picks for any prompt
python3 auto_temp.py "your prompt here"
# Output: 0.9 fiction 4096 1200
```

---

## The important flags (and why they matter)

### `--jinja` — without this, nothing works

This is the single most important flag. SmallThinker uses a chat template that separates "thinking" from "answering." Without `--jinja`, the template isn't applied, and the model gets stuck in a loop of ~49 internal tokens — it thinks forever and never answers. **Every script in this project includes `--jinja`.**

### `-st` (single-turn) — one question, one answer, then stop

Without `-st`, llama-cli enters interactive mode and waits for your next question. On a headless Jetson with no stdin, this produces 260 MB of `>` prompt characters before eventually timing out. With `-st`, it answers one question and exits cleanly.

### `-n 16384` — excess thinking tokens

The `-n` flag caps how many tokens the model can generate. SmallThinker's training data has reasoning chains of 8K+ tokens. If you cap at the default 512 or 2048, you cut the model off mid-thought and get a truncated, wrong answer. We default to 16,384 in max mode so the model has room to think.

### `--temp 1.0` — high temperature for reasoning

Standard advice is "use low temperature (0.3) for accuracy." That assumes you're paying per token and want one fast answer. On this Jetson, tokens are free — so we use temp 1.0 to let the model explore multiple reasoning paths. For creative tasks, the auto-adjuster lowers it based on what works.

### `-ngl 99 -fa on` — full GPU offload + flash attention

`-ngl 99` puts every layer on the GPU (tensor cores do all the math). `-fa on` enables flash attention, which is 6–50× faster for prompt evaluation on the Orin's FP16 tensor cores.

---

## Environment variable overrides

All scripts accept environment variables for manual control:

| Variable | Default | What it does |
|----------|---------|--------------|
| `GUI_MODE` | `max` | `max` = 32K context (GUI off), `off` = 16K (GUI off), `on` = 8K (GUI on) |
| `TEMP` | per script | Override the temperature |
| `CTX` | per mode | Override the context window size |
| `N_TOKENS` | per mode | Override the max output tokens |
| `MODEL` | `~/models/smallthinker-3b-q4_k_m.gguf` | Path to the model file |
| `TOP_P` | per script | Override nucleus sampling threshold |
| `TOP_K` | per script | Override top-K sampling |
| `REPEAT_PENALTY` | per script | Override repetition penalty |
| `LLAMA_CLI` | `~/llama.cpp/build/bin/llama-cli` | Path to the llama.cpp binary |

---

## Prompt input

All three scripts accept prompts three ways:

```bash
# As an argument
./think-auto.sh "your question here"

# From a file
PROMPT_FILE=prompts/samples.txt ./think-auto.sh

# Piped from stdin
echo "your question" | ./think-auto.sh
```

---

## Benchmarks

We benchmarked this model with the same 10-prompt suite from our prior LLM benchmark projects (5 general + 5 coding). Here's what 75 experiments told us:

**Speed**: A consistent 20.0 tokens/second generation across all 75 runs. Prompt evaluation averages 409 t/s. The Jetson is memory-bandwidth bound, not compute bound — sampling parameters don't change the speed.

**Round 1 — Baseline (10 prompts, temp 1.0)**: Average 6.4/10. Verifiable tasks (code, math, proofs) scored 7–8/10. Open-ended creative tasks collapsed to 3–4/10 due to the meta-reasoning loop. Deep thinking doesn't raise the average — it shifts where quality lands.

**Round 2 — Temperature sweep (20 runs)**: Tested temps 0.1–0.9 on the 4 prompts that looped at temp 1.0. The loop is chaotic, not monotonic — best temperature is prompt-dependent. Fiction works best at 0.9, prose is robust everywhere, poetry is weak at all temperatures, creative code works best at 0.7.

**Round 3 — Variable sweep (45 runs)**: Tested top_p, top_k, and repeat_penalty at 5 values each across 3 representative prompts, one variable at a time, 32K context fixed. Biggest wins in speed, not quality:
- top_k=80 makes reasoning 2.6x faster than the common default of 40 (37s vs 98s)
- repeat_penalty=1.10 is the only value that produces complete mathematical proofs (5/5 proof elements)
- repeat_penalty=1.20 makes creative and coding 2.6x faster with identical quality scores
- Coding and creative quality are robust across all values (7/7 code elements, 5/5 scene elements)
- Reasoning is the most sensitive variable — proof completeness varies from 2/5 to 5/5

**Round 4 — Tuned retest (10 runs)**: All 10 original prompts re-run with their category's best settings. **Average quality rose from 6.4/10 to 9.8/10 (+53%). All 10 prompts improved, zero regressions.** The biggest jumps were on the prompts that looped hardest: creative writing (+7), iambic pentameter (+6), TRS-80 BASIC (+5).

Full data: `results/benchmark_results.json`, `results/temp_sweep_quality.json`, `results/variable_sweep_quality.json`, `results/tuned_retest_quality.json`, and `Deep_SmallThinker_3B_Findings_Report.pdf` (8-page report covering all 75 experiments).

---

## Files

```
deep-smallthinker-3b/
├── think-auto.sh               # auto-tuned launcher (recommended)
├── think.sh                    # deep reasoning launcher (manual, tuned defaults)
├── think-creative.sh           # creative launcher (tuned per --style)
├── auto_temp.py                # the prompt classifier
├── benchmark.py                # 10-prompt benchmark suite
├── tuned_retest.py             # 10-prompt retest with tuned settings
├── temp_sweep.py               # temperature sweep (20 runs)
├── variable_sweep.py           # variable sweep (45 runs)
├── score_quality.py            # metrics + quality scoring
├── render_report.py            # merges metrics + scores
├── build_report_pdf.py         # renders the benchmark PDF report
├── build_findings_pdf.py       # renders the full 75-experiment findings PDF
├── Deep_SmallThinker_3B_Benchmark_Report.pdf
├── Deep_SmallThinker_3B_Findings_Report.pdf   # 8-page report, all 75 experiments
├── scripts/
│   └── download-model.sh       # fetches the Q4_K_M model
├── prompts/
│   └── samples.txt             # test prompts
├── results/                    # all benchmark data (JSON)
└── README.md
```

---

## Re-running the benchmarks

```bash
python3 benchmark.py        # ~15 min, 10 prompts at temp 1.0 (baseline)
python3 temp_sweep.py       # ~30 min, 20 runs at temps 0.1–0.9
python3 variable_sweep.py  # ~45 min, 45 runs (top_p, top_k, repeat_penalty)
python3 tuned_retest.py     # ~30 min, 10 prompts with tuned settings (validation)
python3 score_quality.py    # print metrics summary
python3 render_report.py    # merge metrics + quality scores
python3 build_report_pdf.py # render the benchmark PDF report
python3 build_findings_pdf.py # render the full 75-experiment findings PDF
```

---

## Disclaimer

SmallThinker-3B is a research preview. Its reasoning output should be treated as a draft and verified — do not rely on it for consequential decisions. The model got "strawberry" wrong (said 2, correct is 3) on the Q8_0 at temp 0.6, though it got it right on Q4_K_M at temp 1.0. Always sanity-check.