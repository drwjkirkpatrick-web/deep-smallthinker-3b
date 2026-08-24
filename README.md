# deep-smallthinker-3b

A single-command way to run [SmallThinker-3B](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) — a small AI model that "thinks out loud" before it answers — on an 8GB Jetson. You ask one question, it shows you its full chain of thought, then gives you the answer. No setup between questions, no interactive loop, just ask-and-answer.

This project includes something most don't: an **automatic temperature adjuster** that reads your prompt and picks the best settings for it. We tested 20 different temperature-and-prompt combinations to find what works, and the launcher uses those findings automatically.

---

## What is SmallThinker-3B?

It's a 3.4-billion-parameter AI model built on Qwen2.5-3B. Unlike a regular chatbot that answers immediately, SmallThinker writes out its reasoning first — sometimes thousands of words of "let me consider..." and "if X then Y..." before it arrives at a conclusion. Think of it as a model that shows its work on every question.

**What it's good at:** math proofs, coding problems, logic puzzles, step-by-step reasoning — tasks where you can check whether the answer is right.

**What it struggles with:** creative writing, poetry, open-ended prose — tasks where the "thinking" part can spiral into an endless loop of "I need to write X... first I should... let me think..." and never actually produce the thing you asked for. We discovered this the hard way and built the temperature adjuster to fix it.

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

### The easy way — automatic temperature

```bash
./think-auto.sh "A bat and a ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?"
```

That's it. The auto-adjuster reads your prompt, figures out what kind of question it is, and picks the best temperature for it. You'll see:

```
== think-auto.sh (auto temperature) ==
  prompt:    A bat and a ball cost $1.10 total. The bat costs $1 more...
  style:     reasoning
  temp:      1.0  ← auto-selected
  context:   32768
  tokens:    16384
========================================

<thousands of words of chain-of-thought reasoning>

The ball costs $0.05.

[ Prompt: 279.1 t/s | Generation: 20.0 t/s ]
```

### What the auto-adjuster does

It classifies your prompt into one of five categories and uses the best temperature we found for each:

| Your prompt asks for... | Auto-detected as... | Temperature | Why |
|-------------------------|---------------------|-------------|-----|
| Math, proofs, algorithms, logic | reasoning | 1.0 | High temp lets it explore the reasoning tree — and since there's a checkable answer, exploration helps |
| A story, scene, or narrative | fiction | 0.9 | The only temperature where the model commits to actual prose instead of planning forever |
| A poem, sonnet, or verse | poetry | 0.3 | Lower temp keeps the model from spiraling (though poetry is its weakest skill) |
| An explanation or description | prose | 0.5 | Prose is robust at any temperature; 0.5 is the sweet spot for thoroughness |
| A game, retro code, or interactive program | code | 0.7 | Closest to correct output for creative/structured code |

> **How did we find these numbers?** We ran a 20-experiment temperature sweep — 4 prompt types × 5 temperatures (0.1, 0.3, 0.5, 0.7, 0.9) — keeping everything else fixed and changing *only* the temperature. Full data is in `results/temp_sweep_quality.json`.

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

| Script | When to use | Temperature | Context | Tokens |
|--------|-------------|-------------|---------|--------|
| `think-auto.sh` | Default — let it auto-adjust | Auto (0.3–1.0) | Auto | Auto |
| `think.sh` | Deep reasoning, manual control | 1.0 (override with `TEMP=`) | 32K (max mode) | 16K |
| `think-creative.sh` | Creative writing, manual style pick | Per style (0.3–0.9) | 4K | 1.2K |

---

## How the temperature adjuster works

SmallThinker has a quirk: on open-ended creative tasks, it can fall into a **meta-reasoning loop**. Instead of writing the scene you asked for, it writes:

> *"I need to write a scene about a lighthouse. First, I should think about the setting. Let me consider the atmosphere. I'll need a character..."*

...and it does this for the entire 8,000-token budget, never writing the actual scene. This happens at *most* temperatures, and it's **not predictable** — the same prompt at temp 0.3 might work while temp 0.5 loops, and temp 0.9 might be the best for one creative task but the worst for another.

Our 20-run sweep found the best temperature for each task type (see the table in Step 4). The `auto_temp.py` classifier reads your prompt, matches it against keyword patterns for each category, and outputs the winning temperature. It's a simple keyword classifier — no AI needed for the classification itself, just pattern matching.

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

We benchmarked this model with the same 10-prompt suite from our prior LLM benchmark projects (5 general + 5 coding). The key findings:

**Speed**: A consistent 20.0 tokens/second generation across all prompts. Prompt evaluation averages 409 t/s.

**Quality at temp 1.0 (deep thinking)**: Average 6.4/10 — identical to the earlier temp-0.3 run. Deep thinking doesn't raise the average; it shifts where quality lands. Verifiable tasks (code, math, proofs) scored 7–8/10. Open-ended creative tasks collapsed to 3–4/10 due to the meta-reasoning loop.

**Temperature sweep**: A 20-run sweep (temp 0.1–0.9, only temperature varied) found the loop is chaotic, not monotonic. Best temperature is prompt-dependent. Creative fiction works best at 0.9, prose is robust everywhere, poetry is weak at all temperatures, creative code works best at 0.7.

Full benchmark data: `results/benchmark_results.json`, `results/temp_sweep_quality.json`, and `Deep_SmallThinker_3B_Benchmark_Report.pdf`.

---

## Files

```
deep-smallthinker-3b/
├── think-auto.sh               # auto-temperature launcher (recommended)
├── think.sh                    # deep reasoning launcher (manual temp)
├── think-creative.sh           # creative launcher (temp per --style)
├── auto_temp.py                # the temperature classifier
├── benchmark.py                # 10-prompt benchmark suite
├── temp_sweep.py               # temperature sweep (20 runs)
├── score_quality.py            # metrics + quality scoring
├── render_report.py            # merges metrics + scores
├── build_report_pdf.py         # renders the PDF report
├── Deep_SmallThinker_3B_Benchmark_Report.pdf
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
python3 benchmark.py        # ~15 min, 10 prompts at temp 1.0
python3 temp_sweep.py       # ~30 min, 20 runs at temps 0.1–0.9
python3 score_quality.py    # print metrics summary
python3 render_report.py    # merge metrics + quality scores
python3 build_report_pdf.py # render the PDF report
```

---

## Disclaimer

SmallThinker-3B is a research preview. Its reasoning output should be treated as a draft and verified — do not rely on it for consequential decisions. The model got "strawberry" wrong (said 2, correct is 3) on the Q8_0 at temp 0.6, though it got it right on Q4_K_M at temp 1.0. Always sanity-check.