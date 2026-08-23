# deep-smallthinker-3b

Single-turn deep reasoning with [SmallThinker-3B](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) on an 8GB Jetson (Orin Nano). One command, tuned for **maximum thinking depth**: an excess of thinking tokens, a high temperature (we have free tokens, so we let it explore), and a context window sized for GUI-off operation.

## What this is

SmallThinker-3B is a 3.4B-parameter reasoning model (Qwen2.5-3B base) that emits a full chain-of-thought before answering. The problem with running it naively is threefold:

1. **It gets stuck without `--jinja`** — the chat template isn't applied, so it loops forever on internal `<think>` tokens (~49 tokens) and never produces an answer.
2. **Its thinking gets truncated** — default token budgets are far too small to let it reason.
3. **Its default sampling is too conservative** — with time and tokens not being a constraint on this Jetson, a low temperature wastes the model's ability to explore.

This project fixes all three with a single, documented launch script.

## Prerequisites

- Jetson Orin Nano 8GB (JetPack 6, L4T R36.5.x) — or any aarch64/ARM64 box with ≥8GB unified memory
- llama.cpp built with CUDA (see [jetson-llamacpp-benchmarks](https://github.com/drwjkirkpatrick-web/jetson-llamacpp-benchmarks) for the exact CMake flags)
- The Q4_K_M GGUF (~2.0 GB)

```bash
# Build llama.cpp once
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_CUDA_F16=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)

# Download the model (into ~/models/)
./scripts/download-model.sh
```

## Quick start

```bash
# Stop the GUI to free ~600 MB (optional but recommended for the big contexts)
sudo systemctl stop gdm3

# Ask a question — single turn, deep thinking, then exit
./think.sh "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?"
```

Outputs the full chain-of-thought, then the boxed final answer, then a speed summary:
```
[ Prompt: 279.1 t/s | Generation: 19.2 t/s ]
Exiting...
```

## GUI modes

The context window and thinking budget scale with how much RAM you free:

| Mode | Command | Context | Thinking tokens | Est. runtime | Notes |
|------|---------|---------|-----------------|--------------|-------|
| `max` (default) | `./think.sh "..."` | **32768** | **16384** | ~3.6 GB | Native 32K; GUI MUST be off |
| `off` | `GUI_MODE=off ./think.sh "..."` | **16384** | **8192** | ~3.0 GB | GUI off, lighter RAM |
| `on` | `GUI_MODE=on ./think.sh "..."` | 8192 | 4096 | ~2.7 GB | GUI still running |

Memory math (measured on this Jetson, 7.4 GiB usable):
- GUI on → ~4.1 GB free · GUI off → ~4.7 GB free
- Q4_K_M weights: ~2.0 GB
- KV cache (Qwen2.5-3B, f16): ~36 KB/token → 302 MB @ 8K, 604 MB @ 16K, 1.2 GB @ 32K
- CUDA context overhead: ~400 MB

## Settings and rationale

### Excess thinking tokens (`-n`)
SmallThinker was trained to emit 8K+ token reasoning chains (the QwQ-LongCoT-500K dataset is >75% samples with >8K output tokens). Default `-n` values truncate this. This project defaults to **16384** thinking tokens (the `max` mode) so the model can reason as long as it wants on a single turn.

### High temperature (`--temp 1.0`)
The standard advice for reasoning models is a low temperature (0.3–0.6) for determinism. That advice assumes you're paying per token and want one correct answer fast. On this Jetson **tokens are free and time is not a factor**, so we flip it: `--temp 1.0` (the paper's own generation setting) lets the model explore the reasoning tree and re-derive. If you need precision on a math/code problem, override with `TEMP=0.6 ./think.sh "..."`.

### Context (`-c`)
Defaults to the model's **native 32K (32768)** context in `max` mode — a short prompt plus up to 16K of thinking fits comfortably with the GUI stopped. `off` mode drops to 16K, `on` mode to 8K for lighter RAM.

### Single turn (`-st` / `--single-turn`)
`-st` runs exactly one turn then exits — no interactive REPL, no 260 MB of `>` prompts, no hanging stdin. Combined with `--no-conversation` it is clean and scriptable. This is the core UX: *one question in, one deep answer out*.

### The rest
| Flag | Value | Why |
|------|-------|-----|
| `--jinja` | on | **Critical** — applies the chat template so the thinking model actually emits its answer (otherwise it loops on ~49 internal tokens forever) |
| `-ngl 99` | all layers | Offload every layer to the GPU (tensor cores for all matmul) |
| `-fa on` | flash attention | 6–50× faster prompt eval on the Orin's FP16 tensor cores |
| `--top-p` / `--top-k` | 0.95 / 40 | Standard nucleus sampling; keeps reasoning on-track at temp 1.0 |
| `--repeat-penalty` | 1.1 | Discourages the model from re-stating itself in long CoT loops |
| `-t 6` | 6 threads | Matches the Orin's 6 CPU cores |
| `--no-display-prompt` | — | Clean output: answer only, no prompt echo |

## Environment variable overrides

| Var | Default | Effect |
|-----|---------|--------|
| `GUI_MODE` | `off` | `off` / `on` / `max` → context + token budget |
| `CTX` | per mode | Explicit context override |
| `N_TOKENS` | per mode | Explicit thinking-token override |
| `TEMP` | `1.0` | Sampling temperature |
| `TOP_P` / `TOP_K` | `0.95` / `40` | Nucleus sampling |
| `REPEAT_PENALTY` | `1.1` | Repetition penalty |
| `MODEL` | `~/models/smallthinker-3b-q4_k_m.gguf` | Model path |
| `LLAMA_CLI` | `~/llama.cpp/build/bin/llama-cli` | Binary path |

## Prompt input

Three ways to supply the prompt:

```bash
./think.sh "your question"              # as an argument
PROMPT_FILE=prompts/samples.txt ./think.sh  # from a file
echo "your question" | ./think.sh       # piped on stdin
```

## Verification

```bash
# Confirm the model loads and answers (fast smoke test)
./think.sh "Count the r's in strawberry."
# Expect: a chain-of-thought trace, a final boxed answer, then the speed summary.
# (It may or may not get the count right — that is the point; see note below.)
```

> Note: "strawberry" has **3** r's. SmallThinker is a 3B model — its chain-of-thought is a *draft*, not ground truth. It scores 8/10 on math in our Jetson benchmark but only 6.4/10 overall; always sanity-check its final answer against your own reasoning.

## Benchmarks (this Jetson, August 2026)

| Quant | Size | Gen tok/s | Prompt tok/s |
|-------|------|-----------|--------------|
| Q8_0 | 3.37 GB | 19.2 | 279 |
| Q4_K_M | 2.1 GB | **20–21** | 172–309 |

Q4_K_M is chosen for this project because the smaller weights free ~1.3 GB, which is exactly what buys the larger context windows (16K–32K) for deep thinking. Measured on this Jetson, August 2026: the Q4_K_M model correctly solved both the "strawberry r-count" (3) and the bat-and-ball ($0.05) problems with the deep-thinking settings at temp 1.0.

## Files

```
deep-smallthinker-3b/
├── think.sh                 # main launch script (single-turn deep reasoning)
├── scripts/
│   └── download-model.sh    # fetches the Q4_K_M GGUF
├── prompts/
│   └── samples.txt          # reasoning test prompts (traps, math, logic, code)
└── README.md
```

## Disclaimer

SmallThinker-3B is a research preview. Its reasoning output should be treated as a draft and verified — do not rely on it for consequential decisions.
