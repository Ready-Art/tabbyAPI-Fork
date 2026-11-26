# TabbyAPI

<p align="left">
    <img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue" alt="Python 3.10, 3.11, and 3.12">
    <a href="/LICENSE">
        <img src="https://img.shields.io/badge/License-AGPLv3-blue.svg" alt="License: AGPL v3"/>
    </a>
    <a href="https://discord.gg/sYQxnuD7Fj">
        <img src="https://img.shields.io/discord/545740643247456267.svg?logo=discord&color=blue" alt="Discord Server"/>
    </a>
</p>

<p align="left">
    <a href="https://theroyallab.github.io/tabbyAPI">
        <img src="https://img.shields.io/badge/Documentation-API-orange" alt="Developer facing API documentation">
    </a>
</p>

<p align="left">
    <a href="https://ko-fi.com/I2I3BDTSW">
        <img src="https://img.shields.io/badge/Support_on_Ko--fi-FF5E5B?logo=ko-fi&style=for-the-badge&logoColor=white" alt="Support on Ko-Fi">
    </a>
</p>

> [!IMPORTANT]
>
> In addition to the README, please read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/1.-Getting-Started) page for information about getting started!

> [!NOTE]
> 
> Need help? Join the [Discord Server](https://discord.gg/sYQxnuD7Fj) and get the `Tabby` role. Please be nice when asking questions.

> [!NOTE]
>
> Tool calling support has been revamped and now no longer relies on modified Jinja templates. [See the docs for more.](docs/10.-Tool-Calling.md)

> [!NOTE]
>
> This fork (branch `ReadyArt`) adds support for the [ReadyArt ExLlamaV2 fork](https://github.com/Ready-Art/exllamav2-Fork): single-process (`sp`) and multiprocess (`mp`) tensor parallelism, MTP speculative decoding on both, TP CUDA graphs, FlashInfer attention (required by the fork) plus Flash-Attention 2 (also required for now), and Gemma-4 thinking/tool-call handling incl. `reasoning_effort`. [Setup and configuration docs.](docs/11.-ReadyArt-ExLlamaV2-Fork.md)

> [!NOTE]
> 
> Want to run GGUF models? Take a look at [YALS](https://github.com/theroyallab/YALS), TabbyAPI's sister project.

A FastAPI based application that allows for generating text using an LLM (large language model) using the [Exllamav2](https://github.com/turboderp-org/exllamav2) and [Exllamav3](https://github.com/turboderp-org/exllamav3) backends.

TabbyAPI is also the official API backend server for ExllamaV2 and V3.

**Multi-GPU / MTP:** install this fork with the [ReadyArt ExLlamaV2 fork](https://github.com/Ready-Art/exllamav2-Fork) for tensor parallelism (SP/MP), MTP speculative decoding, CUDA graphs and Gemma-4 thinking/tool calls — see [Installing with the ReadyArt ExLlamaV2 fork](#installing-with-the-readyart-exllamav2-fork-multi-gpu--mtp) below.

## Disclaimer

This project is marked as rolling release. There may be bugs and changes down the line. Please be aware that you might need to reinstall dependencies if needed.

TabbyAPI is a hobby project made for a small amount of users. It is not meant to run on production servers. For that, please look at other solutions that support those workloads.

## Getting Started

> [!IMPORTANT]
> 
> Looking for more information? Check out the Wiki.

For a step-by-step guide, choose the format that works best for you:

📖 Read the [Wiki](https://github.com/theroyallab/tabbyAPI/wiki/01.-Getting-Started) – Covers installation, configuration, API usage, and more.

🎥 Watch the [Video Guide](https://www.youtube.com/watch?v=03jYz0ijbUU) – A hands-on walkthrough to get you up and running quickly.

## Installing with the ReadyArt ExLlamaV2 fork (multi-GPU / MTP)

This fork ([Ready-Art/tabbyAPI-Fork](https://github.com/Ready-Art/tabbyAPI-Fork), branch `ReadyArt`)
supports the [ReadyArt ExLlamaV2 fork](https://github.com/Ready-Art/exllamav2-Fork): single-process (`sp`)
and multiprocess (`mp`) tensor parallelism, MTP speculative decoding on both, TP CUDA graphs, the
**EXL2.5 quantization format**, and Gemma-4 thinking / tool calls.

> **Scope.** Built and tested on **SM86 (Ampere — RTX 3090)**, end-to-end on **Gemma-4-31B-IT** only.
> **Qwen3.8-27B** is a **work in progress and not yet usable** (base decode made coherent, but
> eager hybrid decode, broken built-in MTP + multi-batch, and gen tok/s far below target).
> **Qwen3.8-Flash-Next** and **Gemma-4-26B-A4B-it** are **not yet implemented** (planned);
> see [docs/11](docs/11.-ReadyArt-ExLlamaV2-Fork.md).
> The primary, optimized path is **MP + CUDA graphs + MTP** on **4× RTX 3090, PCIe P2P
> (`NCCL_P2P_LEVEL=PXB`), 200 W power cap per card** — reported numbers are a 200 W-cap lower
> bound; full-powered GPUs are significantly faster. `sp`, `native`, non-MTP and other model
> families work but are secondary. Details, caveats and the full feature list:
> [docs/11.-ReadyArt-ExLlamaV2-Fork.md](docs/11.-ReadyArt-ExLlamaV2-Fork.md) and the
> [exllamav2-Fork README](https://github.com/Ready-Art/exllamav2-Fork).
> Issues and PRs welcome on either repo, against the `ReadyArt` branch.

**Requirements:** Python 3.12, a CUDA build of PyTorch with a CUDA toolkit (`nvcc` on `PATH`),
**FlashInfer >= 0.7.4** (a hard requirement of the fork), and **Flash-Attention 2 — required for now
even with FlashInfer** (the fork's TP / dynamic-generator paths still call flash-attn kernels).
Tested combination: Python 3.12.15, torch 2.15.0+cu132, CUDA toolkit 13.5, FlashInfer 0.7.4,
flash-attn 2.8.3. Source installs of the fork are detected automatically and work with CUDA 13 Torch
(prebuilt upstream exllamav2 wheels are CUDA 12 only).

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python — it installs
Python 3.12 itself, so no system Python 3.12 package is needed:

```bash
# 0. uv + Python 3.12 environment + CUDA 13.2 torch (tested combination).
#    Install torch BEFORE building the fork: its setup.py compiles against the installed torch.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv venv venv --python 3.12
source venv/bin/activate
uv pip install torch==2.15.0+cu132 --index-url https://download.pytorch.org/whl/cu132

# 1. TabbyAPI (this fork)
git clone https://github.com/Ready-Art/tabbyAPI-Fork
cd tabbyAPI-Fork
uv pip install -e .

# 2. The ReadyArt exllamav2 fork, built from source for your GPU.
#    TORCH_CUDA_ARCH_LIST: "8.6" for RTX 3090, "8.9" for RTX 4090, etc.
git clone https://github.com/Ready-Art/exllamav2-Fork
cd exllamav2-Fork
uv pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST="8.6" uv pip install -e . --no-build-isolation
cd ..

# 3. FlashInfer kernels (hard requirement; kernel wheels are CUDA/torch-version specific —
#    re-run download-kernels after every FlashInfer upgrade)
flashinfer download-kernels

# 4. Flash-Attention 2 — required for now even with FlashInfer.
#    Wheel must match Python/torch/CUDA; this one is for Python 3.12 / torch 2.15 / CUDA 13.2.
#    Other combinations: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v1.5.0/flash_attn-2.8.3+cu132torch2.15-cp312-cp312-linux_x86_64.whl"

# 5. Model + MTP drafter
hf download ReadyArt/Gemma-4-31B-IT-EXL2-6.0BPW-HB16 \
    --local-dir models/Gemma-4-31B-IT-EXL2-6.0BPW-HB16
hf download google/gemma-4-31B-it-assistant \
    --local-dir models/gemma-4-31B-it-assistant
```

Example `config.yml` — **the configuration used on the tested 4× RTX 3090 rig** (MP tensor
parallelism + MTP drafting, 160k context, Gemma-4 thinking and tool calls enabled). Model paths
are shown relative (`models/`); point them at your own model directory:

```yml
network:
  host: 127.0.0.1                 # 0.0.0.0 to accept remote clients
  port: 5000
  disable_auth: false

memory:
  cuda_malloc_async: true         # NOTE: the fork TP backends force the native allocator
                                  # regardless (cudaMallocAsync corrupts TP CUDA-graph replay)

model:
  model_dir: models
  model_name: Gemma-4-31B-IT-EXL2-6.0BPW-HB16
  backend: exllamav2
  max_seq_len: 160000
  cache_size: 160000            # per-row KV depth; the ~68 tok/s recipe below uses 32768
  chunk_size: 2048
  tensor_parallel: true
  tensor_parallel_backend: mp     # one worker process per GPU (the primary tested path)
  gpu_split: [22, 22, 22, 22]     # GiB per GPU (4× 24 GB RTX 3090)
  tp_graphs: true                 # whole-step TP CUDA graphs
  reasoning: true                 # Gemma-4 thought-channel tags are auto-detected
  tool_format: gemma4             # also auto-detected for Gemma-4

draft_model:
  draft_model_dir: models
  draft_mode: mtp
  draft_model_name: gemma-4-31B-it-assistant
  draft_num_tokens: 3             # MTP verify window width (3 = measured best on prose)
  draft_device: 0
```

**Launch the server with P2P enabled** (this is how the tested rig runs it):

```bash
NCCL_P2P_LEVEL=PXB EXL2_MTP_DRAFT_GRAPH=1 python main.py
```

With `draft_num_tokens: 3` and the whole-window drafter graph enabled
(`EXL2_MTP_DRAFT_GRAPH=1`), this measures **~68 tok/s generation** on Gemma-4-31B-IT at
~1.3k context (temperature-1 sampling, 4× RTX 3090 power-limited to 200 W, `cache_size: 32768`,
`chunk_size: 2048`, `gpu_split: [22, 22, 22, 22]`, `max_batch_size: 1`; 2026-11-21).

**EXL2.5 models:** checkpoints converted with the fork's EXL2.5 format (uniform-bit fast path,
optional Hadamard/AWQ) load automatically — the format tags are self-describing in the
safetensors, so no config switch is needed; just point `model_name` at the directory. The
fork's CUDA extension must be built from the `ReadyArt` branch (EXL2.5 tensors are served by
its fast-path kernels; had/awq tensors cannot run on the legacy kernel). Plain EXL2 models are
unaffected. See the
[exllamav2-Fork README](https://github.com/Ready-Art/exllamav2-Fork#exl25-quantization-format)
for the format rules and conversion flags.

`NCCL_P2P_LEVEL=PXB` enables PCIe peer-to-peer across the PCI switch on the tested 4× RTX 3090
box. **The right value depends on your topology** — `PXB`/`PHB`/`SYS`/`NODE` describe how far
apart the GPUs are (see the NCCL docs); on a NVLink rig you would not need this. The fork's custom
P2P collective kernel also requires peer access between GPUs (it probes `canAccessPeer` and falls
back to NCCL automatically if unavailable). Then point any OpenAI-compatible client at
`http://localhost:5000/v1`. Thinking is controlled per request via
`chat_template_kwargs: {"enable_thinking": true}` or `reasoning_effort`
(`none`/`minimal` off, `low`/`medium`/`high` on).

Notes and limitations (batch size 1 by default on the fork TP/MTP paths — **experimental
multi-batch** opt-in via `max_batch_size` on MP + MTP, validated under concurrent load at
`max_batch_size: 4` with per-slot prompt caching and shared-prefix KV cloning; **experimental
prompt-cache RAM offload** opt-in via `prompt_cache_offload` + `prompt_cache_ram_limit_mb`,
tiering idle conversations to host RAM under a **whole-server** byte cap so the VRAM context
budget is not a hard limit on cached conversations when sharing the API; **INT8 KV cache**
opt-in via `kv_cache_int8` (default fp16) — INT8-primary storage (no fp16 shadow) halving
KV VRAM and long-context decode/verify KV reads; **priority
prompt scheduler** opt-in via `priority_queue` — shortest-first admission plus a
reprocess-abuse penalty for shared-API serving; **engram storage modes** `ngram_mode`
disk/ram/hybrid for n-gram models; full-sampler
MTP on both SP and MP, no LoRAs on MP, CUDA graphs and FlashInfer knobs) are documented in
**[docs/11.-ReadyArt-ExLlamaV2-Fork.md](docs/11.-ReadyArt-ExLlamaV2-Fork.md)**.

**DFlash (experimental):** the fork also supports
[z-lab's DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash) block-diffusion drafter —
a whole block (anchor + up to 15 masks) drafted in one parallel pass, FlashInfer attention.
Set `draft_mode: dflash` with `draft_model_name: gemma-4-31B-it-DFlash` (MP backend,
`max_batch_size: 1`; larger windows favor DFlash). See the
[DFlash section](docs/11.-ReadyArt-ExLlamaV2-Fork.md#dflash-block-diffusion-drafting-experimental-mp)
of the fork docs.

## Features

- OpenAI compatible API
- Loading/unloading models
- HuggingFace model downloading
- Embedding model support
- JSON schema + Regex + EBNF support
- AI Horde support
- Speculative decoding via draft models
- Multi-lora with independent scaling (ex. a weight of 0.9)
- Inbuilt proxy to override client request parameters/samplers
- Flexible Jinja2 template engine for chat completions that conforms to HuggingFace
- Concurrent inference with asyncio
- Utilizes modern python paradigms
- Continuous batching engine using paged attention
- Fast classifier-free guidance
- OAI style tool/function calling

And much more. If something is missing here, PR it in!

## Supported Model Types

TabbyAPI uses Exllama as a powerful and fast backend for model inference, loading, etc. Therefore, the following types of models are supported:

- Exl2/GPTQ (deprecated, will be removed in the near future)

- Exl3 (Highly recommended)

- FP16/BF16

In addition, TabbyAPI supports parallel batching using paged attention for Nvidia Ampere GPUs and higher.

## Contributing

Use the template when creating issues or pull requests, otherwise the developers may not look at your post.

If you have issues with the project:

- Describe the issue in detail

- If you have a feature request, please indicate it as such.

If you have a Pull Request

- Describe the pull request in detail, what, and why you are changing something

## Acknowldgements

TabbyAPI would not exist without the work of other contributors and FOSS projects:

- [ExllamaV2](https://github.com/turboderp-org/exllamav2)
- [ExllamaV3](https://github.com/turboderp-org/exllamav3)
- [Aphrodite Engine](https://github.com/PygmalionAI/Aphrodite-engine)
- [infinity-emb](https://github.com/michaelfeil/infinity)
- [FastAPI](https://github.com/fastapi/fastapi)
- [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)
- [SillyTavern](https://github.com/SillyTavern/SillyTavern)

## Developers and Permissions

Creators/Developers:

- [kingbri](https://github.com/bdashore3)

- [Splice86](https://github.com/Splice86)

- [Turboderp](https://github.com/turboderp)
