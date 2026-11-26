from pydantic import (
    BaseModel,
    ConfigDict,
    constr,
    Field,
    PrivateAttr,
    field_validator,
)
from typing import List, Literal, Optional, Union


CACHE_SIZES = Literal["FP16", "Q8", "Q6", "Q4"]
CACHE_TYPE = Union[CACHE_SIZES, constr(pattern=r"^[2-8]\s*,\s*[2-8]$")]


class Metadata(BaseModel):
    """metadata model for config options"""

    include_in_config: Optional[bool] = Field(True)


class BaseConfigModel(BaseModel):
    """Base model for config models with added metadata"""

    _metadata: Metadata = PrivateAttr(Metadata())


class ConfigOverrideConfig(BaseConfigModel):
    """Model for overriding a provided config file."""

    # TODO: convert this to a pathlib.path?
    config: Optional[str] = Field(None, description="Path to an overriding config.yml file")

    _metadata: Metadata = PrivateAttr(Metadata(include_in_config=False))


class NetworkConfig(BaseConfigModel):
    """Options for networking"""

    host: Optional[str] = Field(
        "127.0.0.1",
        description=(
            "The IP to host on (default: 127.0.0.1).\n"
            "Use 0.0.0.0 to expose on all network adapters."
        ),
    )
    port: Optional[int] = Field(5000, description="The port to host on (default: 5000).")
    disable_auth: Optional[bool] = Field(
        False,
        description=(
            "Disable HTTP token authentication with requests.\n"
            "WARNING: This will make your instance vulnerable!\n"
            "Turn on this option if you are ONLY connecting from localhost."
        ),
    )
    disable_fetch_requests: Optional[bool] = Field(
        False,
        description=(
            "Disable fetching external content in response to requests,such as images from URLs."
        ),
    )
    send_tracebacks: Optional[bool] = Field(
        False,
        description=(
            "Send tracebacks over the API (default: False).\n"
            "NOTE: Only enable this for debug purposes."
        ),
    )
    api_servers: Optional[List[Literal["oai", "kobold"]]] = Field(
        ["OAI"],
        description=(
            'Select API servers to enable (default: ["OAI"]).\nPossible values: OAI, Kobold.'
        ),
    )

    # Converts all strings in the api_servers list to lowercase
    # NOTE: Expand if more models need this validator
    @field_validator("api_servers", mode="before")
    def api_server_validator(cls, api_servers):
        return [server_name.lower() for server_name in api_servers]


# TODO: Migrate config.yml to have the log_ prefix
# This is a breaking change.
class LoggingConfig(BaseConfigModel):
    """Options for logging"""

    log_prompt: Optional[bool] = Field(
        False,
        description="Enable prompt logging (default: False).",
    )
    log_generation_params: Optional[bool] = Field(
        False,
        description="Enable generation parameter logging (default: False).",
    )
    log_requests: Optional[bool] = Field(
        False,
        description=(
            "Enable request logging (default: False).\nNOTE: Only use this for debugging!"
        ),
    )
    log_chat_completion_requests: Optional[bool] = Field(
        False,
        description=(
            "Write every /v1/chat/completions request to logs/debug/ as JSON (default: False).\n"
            "PRIVACY WARNING: Enabling this creates a comprehensive request log, including the "
            "full message history and generation parameters. API keys are redacted, but prompts "
            "and user-provided content are preserved for bug-report reproduction."
        ),
    )
    log_file: Optional[str] = Field(
        None,
        description=(
            "Mirror ALL console output (stdout + stderr) to this file while keeping the\n"
            "terminal output. Unlike the logs/ loguru sink, this also captures spawned TP\n"
            "worker tracebacks ([rankN] worker error), print() output and native CUDA/NCCL\n"
            "crash messages, because workers inherit the tee'd descriptors. Appends, no\n"
            "rotation (default: None = off). Example: logs/server.log"
        ),
    )


class ModelConfig(BaseConfigModel):
    """
    Options for model overrides and loading
    Please read the comments to understand how arguments are handled
    between initial and API loads
    """

    # TODO: convert this to a pathlib.path?
    model_dir: str = Field(
        "models",
        description=(
            "Directory to look for models (default: models).\n"
            "Windows users, do NOT put this path in quotes!"
        ),
    )
    inline_model_loading: Optional[bool] = Field(
        False,
        description=(
            "Allow direct loading of models "
            "from a completion or chat completion request (default: False).\n"
            "This method of loading is strict by default.\n"
            "Enable dummy models to add exceptions for invalid model names."
        ),
    )
    use_dummy_models: Optional[bool] = Field(
        False,
        description=(
            "Sends dummy model names when the models endpoint is queried. "
            "(default: False)\n"
            "Enable this if the client is looking for specific OAI models.\n"
        ),
    )
    dummy_model_names: List[str] = Field(
        default=["gpt-3.5-turbo"],
        description=(
            "A list of fake model names that are sent via the /v1/models endpoint. "
            '(default: ["gpt-3.5-turbo"])\n'
            "Also used as bypasses for strict mode if inline_model_loading is true."
        ),
    )
    model_name: Optional[str] = Field(
        None,
        description=(
            "An initial model to load.\n"
            "Make sure the model is located in the model directory!\n"
            "REQUIRED: This must be filled out to load a model on startup."
        ),
    )
    use_as_default: List[str] = Field(
        default_factory=list,
        description=(
            "Names of args to use as a fallback for API load requests (default: []).\n"
            "For example, if you always want cache_mode to be Q4 "
            'instead of on the inital model load, add "cache_mode" to this array.\n'
            "Example: ['max_seq_len', 'cache_mode']."
        ),
    )
    backend: Optional[str] = Field(
        None,
        description=(
            "Backend to use for this model (auto-detect if not specified)\n"
            "Options: exllamav2, exllamav3"
        ),
    )
    max_seq_len: Optional[int] = Field(
        None,
        description=(
            "Max sequence length (default: 4096).\nSet to -1 to fetch from the model's config.json"
        ),
        ge=-1,
    )
    cache_size: Optional[int] = Field(
        None,
        description=(
            "Size of the prompt cache to allocate (default: max_seq_len).\n"
            "Must be a multiple of 256 and can't be less than max_seq_len.\n"
            "For CFG, set this to 2 * max_seq_len."
        ),
        multiple_of=256,
        gt=0,
    )
    cache_mode: Optional[CACHE_TYPE] = Field(
        "FP16",
        description=(
            "Enable different cache modes for VRAM savings (default: FP16).\n"
            f"Possible values for exllamav2: {str(CACHE_SIZES)[15:-1]}.\n"
            "For exllamav3, specify the pair k_bits,v_bits where k_bits and v_bits "
            "are integers from 2-8 (i.e. 8,8)."
        ),
    )
    tensor_parallel: Optional[bool] = Field(
        False,
        description=(
            "Load model with tensor parallelism (default: False).\n"
            "Falls back to autosplit if GPU split isn't provided.\n"
            "This ignores the gpu_split_auto value."
        ),
    )
    tensor_parallel_backend: Optional[str] = Field(
        "native",
        description=(
            "Sets a backend type for tensor parallelism. (default: native).\n"
            "Options: native, nccl\n"
            "Native is recommended for PCIe GPUs\n"
            "NCCL is recommended for NVLink.\n"
            "ExllamaV2 (ReadyArt fork) additionally supports: sp, mp\n"
            "sp: single-process TP with the streaming generator (required for\n"
            "sliding-window models like Gemma-4) and optional MTP drafting.\n"
            "mp (alias: nccl): one worker process per GPU, optional MTP drafting."
        ),
    )
    tp_graphs: Optional[bool] = Field(
        True,
        description=(
            "Capture TP decode steps into CUDA graphs (default: True).\n"
            "ExllamaV2 ReadyArt fork only (sp/mp backends).\n"
            "Sets EXL2_TP_GRAPHS. Disable to fall back to eager TP decode."
        ),
    )
    flashinfer_plan_cache: Optional[bool] = Field(
        False,
        description=(
            "Cache FlashInfer attention plans keyed on shape (default: False).\n"
            "ExllamaV2 ReadyArt fork only. Sets EXLLAMA_FLASHINFER_PLAN_CACHE.\n"
            "Experimental: re-planning on every call is the correctness-first default."
        ),
    )
    gpu_split_auto: Optional[bool] = Field(
        True,
        description=(
            "Automatically allocate resources to GPUs (default: True).\n"
            "Not parsed for single GPU users."
        ),
    )
    autosplit_reserve: List[float] = Field(
        [96],
        description=(
            "Reserve VRAM used for autosplit loading (default: 96 MB on GPU 0).\n"
            "Represented as an array of MB per GPU."
        ),
    )
    gpu_split: List[float] = Field(
        default_factory=list,
        description=(
            "Array of VRAM sizes to split between GPUs, in GB (default: []).\n"
            "Used with tensor parallelism."
        ),
    )
    rope_scale: Optional[float] = Field(
        1.0,
        description=(
            "Rope scale (default: 1.0).\n"
            "Same as compress_pos_emb.\n"
            "Use if the model was trained on long context with rope.\n"
            "Leave blank to pull the value from the model."
        ),
    )
    rope_alpha: Optional[Union[float, Literal["auto"]]] = Field(
        None,
        description=(
            "Rope alpha (default: None).\n"
            'Same as alpha_value. Set to "auto" to auto-calculate.\n'
            "Leaving this value blank will either pull from the model "
            "or auto-calculate."
        ),
    )
    chunk_size: Optional[int] = Field(
        2048,
        description=(
            "Chunk size for prompt ingestion (default: 2048).\n"
            "A lower value reduces VRAM usage but decreases ingestion speed.\n"
            "NOTE: Effects vary depending on the model.\n"
            "An ideal value is between 512 and 4096."
        ),
        gt=0,
    )
    output_chunking: Optional[bool] = Field(
        True,
        description=(
            "Use output chunking (default: True)\n"
            "Instead of allocating cache space for the entire completion at once, "
            "allocate in chunks as needed.\n"
            "Used by EXL3 models only.\n"
        ),
    )
    max_batch_size: Optional[int] = Field(
        None,
        description=(
            "Set the maximum number of generation jobs that can run concurrently\n"
            "The default maximum batch size for transformer architectures is 32. Recurrent\n"
            "models with linear or sliding attention use more VRAM to support larger batches,\n"
            "so the default value is reduced to 4. If you do not require concurrency at all, you\n"
            "can reduce it further to minimize VRAM overhead."
        ),
        ge=1,
    )
    prompt_cache_offload: Optional[bool] = Field(
        False,
        description=(
            "Offload idle prompt-cache lineages to host RAM (default: False).\n"
            "ExllamaV2 ReadyArt fork only: mp tensor-parallel backend + MTP drafting\n"
            "(single-sequence and multi-batch). The VRAM cache rows keep the hot\n"
            "lineages; when a new job needs a row, the oldest/least-used idle lineage\n"
            "is copied to RAM instead of being discarded, and a later request matching\n"
            "it restores the K/V (plus the MTP draft shadow) in milliseconds instead of\n"
            "re-prefilling the whole context. RAM usage is capped by\n"
            "prompt_cache_ram_limit_mb; the coldest entries are evicted first when the\n"
            "cap is reached. DFlash drafting is not supported (the option is ignored).\n"
            "EXPERIMENTAL."
        ),
    )
    prompt_cache_ram_limit_mb: Optional[int] = Field(
        8192,
        description=(
            "Host RAM cap for offloaded prompt-cache lineages, in MB (default: 8192).\n"
            "Only used when prompt_cache_offload is enabled. When the store is full,\n"
            "the oldest/least-used lineages are dropped (a later request re-prefills\n"
            "them). The cap is enforced before every offload; the cache system never\n"
            "allocates VRAM and never exceeds this budget. EXPERIMENTAL."
        ),
        ge=256,
    )
    prompt_template: Optional[str] = Field(
        None,
        description=(
            "Set the prompt template for this model. (default: None)\n"
            "If empty, attempts to look for the model's chat template.\n"
            "If a model contains multiple templates in its tokenizer_config.json,\n"
            "set prompt_template to the name of the template you want to use.\n"
            "NOTE: Only works with chat completion message lists!"
        ),
    )
    vision: Optional[bool] = Field(
        False,
        description=("Enables vision support if the model supports it. (default: False)"),
    )
    force_enable_thinking: bool = Field(
        False,
        description=(
            "Force-enable reasoning in template args\n"
            "Injects the enable_thinking: True into the model's template arguments. This doesn't\n"
            "force reasoning or affect how reasoning content is parsed, but some clients will\n"
            "not explicitly enable this and some models need it to properly enter reasoning mode."
        ),
    )
    force_thought_block: bool = Field(
        False,
        description=(
            "Prefill the reasoning opener so the model must think (default: False).\n"
            "Appends the reasoning start token (+ suppress_header, e.g. Gemma-4's\n"
            "'<|channel>thought\\n') to the prompt when reasoning is enabled and the template\n"
            "did not already open the channel. Many finetunes do not spontaneously emit the\n"
            "thought opener; prefilling it forces them into the reasoning channel. Requires\n"
            'reasoning: true. Skipped when thinking is disabled or the opener is already present.'
        ),
    )
    reasoning: bool = Field(
        False,
        description=(
            "Enable the reasoning parser (default: False).\n"
            "Split response message into reasoning_content and content fields."
        ),
    )
    reasoning_start_token: str = Field(
        "<think>",
        description="Start token for the reasoning parser (default: <think>).",
    )
    reasoning_end_token: str = Field(
        "</think>",
        description="End token for the reasoning parser (default: </think>).",
    )
    reasoning_suppress_header: str = Field(
        None,
        description=(
            "Suppress this text whenever it appears in the beginning of a reasoning block "
            "(default: None)."
        ),
    )
    tool_format: Optional[str] = Field(
        None,
        description=(
            "Tool format, e.g. 'qwen3_coder'. See docs for supported formats. If left blank, \n"
            "tool calls from the model will not be parsed by the server."
        ),
    )

    kv_cache_int8: Optional[bool] = Field(
        False,
        description=(
            "INT8 PRIMARY KV cache (default: False = fp16).\n"
            "ExllamaV2 ReadyArt fork only. Sets EXL2_KV_INT8. Ampere has no FP8, so\n"
            "K/V are symmetric per-(token,head) INT8 with fp16 scales, read by a\n"
            "custom multi-warp split-KV attention kernel (decode AND MTP verify).\n"
            "INT8 is the ONLY full-depth KV storage on the full-attention layers —\n"
            "no fp16 shadow alongside — halving KV VRAM (~-50% vs fp16). fp16\n"
            "full-depth buffers exist only as transient prefill staging (freed when\n"
            "prefill ends); SWA rings stay fp16. With MTP, the drafter's joined KV\n"
            "shadow goes INT8 too (half the gather pipe bytes). MP backend, full\n"
            "layers with head_dim 256/512; models with logit softcapping stay fp16."
        ),
    )
    moe_offload_experts: Optional[int] = Field(
        0,
        description=(
            "MoE expert CPU/RAM offload BY EXPERT COUNT (default: 0 = off).\n"
            "ExllamaV2 ReadyArt fork only. Sets EXL2_MOE_OFFLOAD. The last N experts\n"
            "of every MoE layer live in pinned host RAM; one shared GPU staging slot\n"
            "per layer is H2D-filled on demand when routing selects them. N >= the\n"
            "model's expert count offloads ALL experts. Forces the per-expert torch\n"
            "loop (the fused MoE kernel needs every expert resident) — slower decode,\n"
            "much lower VRAM. Single-device load path (no TP sharding) in this release."
        ),
    )
    priority_queue: Optional[bool] = Field(
        False,
        description=(
            "Shortest-prompt-first admission for MP+MTP multi-batch (default: False).\n"
            "ExllamaV2 ReadyArt fork only, requires max_batch_size > 1. Large prefills\n"
            "stall concurrent generation, so short prompts are admitted first and long\n"
            "prompts last. A small batching window (batch_window_ms) collects arriving\n"
            "short prompts so they can be admitted together into the batch slots.\n"
            "priority_max_wait_ms ages long prompts forward to bound starvation.\n"
            "Also enables the reprocess-abuse penalty (reprocess_* options)."
        ),
    )
    batch_window_ms: Optional[int] = Field(
        50,
        description=(
            "Micro-batch admission window in milliseconds (default: 50).\n"
            "When priority_queue is on, an idle batch waits up to this long for more\n"
            "arrivals before admitting, so short prompts arriving alongside a long\n"
            "prompt go first and batch together. Keep small to avoid frustrating API\n"
            "latency. 0 disables the window (pure shortest-first ordering)."
        ),
    )
    batch_window_max_tokens: Optional[int] = Field(
        4096,
        description=(
            "Prompt-length threshold for the batching window (default: 4096 tokens).\n"
            "Prompts at or below this length are 'short' for scheduling purposes."
        ),
    )
    priority_max_wait_ms: Optional[int] = Field(
        30000,
        description=(
            "Starvation guard for priority_queue (default: 30000 ms).\n"
            "A pending prompt that has waited longer than this is promoted ahead of\n"
            "shorter prompts, so a continuous stream of short requests cannot block\n"
            "a long prompt indefinitely."
        ),
    )
    reprocess_min_tokens: Optional[int] = Field(
        8192,
        description=(
            "Reprocess-abuse penalty threshold (default: 8192 tokens).\n"
            "With priority_queue on, prompts at least this long that FULLY re-prefill\n"
            "from scratch (cache miss — not a continuation or re-rendered last reply)\n"
            "are tracked by prompt fingerprint. Repeated full reprocesses of the same\n"
            "long document are deprioritized behind well-behaved traffic."
        ),
    )
    reprocess_strikes: Optional[int] = Field(
        2,
        description=(
            "Full-reprocess strikes before deprioritization (default: 2).\n"
            "The first full prefill of a long document is normal; from this many full\n"
            "reprocesses within reprocess_window_s, repeats of that fingerprint rank\n"
            "last and are held briefly before admission."
        ),
    )
    reprocess_window_s: Optional[int] = Field(
        300,
        description=(
            "Strike counting window in seconds (default: 300).\n"
            "Full-reprocess strikes older than this expire."
        ),
    )
    reprocess_hold_ms: Optional[int] = Field(
        5000,
        description=(
            "Admission hold for flagged reprocess abusers (default: 5000 ms).\n"
            "Flagged prompts wait this long before admission so well-behaved arrivals\n"
            "can jump ahead. reprocess_max_wait_ms overrides the hold."
        ),
    )
    reprocess_max_wait_ms: Optional[int] = Field(
        120000,
        description=(
            "Starvation guard for flagged abusers (default: 120000 ms).\n"
            "Longer than priority_max_wait_ms — delays for abusers are intentional,\n"
            "but they still eventually run."
        ),
    )
    ngram_mode: Optional[str] = Field(
        "disk",
        description=(
            "Engram (hashed n-gram embedding) table storage (default: disk).\n"
            "ExllamaV2 ReadyArt fork only, for models with engram tables (e.g.\n"
            "Qwen3.8-Flash-Next). 'disk': pread rows from the safetensors file on\n"
            "SSD/NVMe per lookup (no host RAM cost). 'ram': full table resident in\n"
            "host RAM (fastest, can be tens of GB). 'hybrid': disk + a bounded LRU\n"
            "row cache in host RAM (ngram_hybrid_mb) — hot n-grams serve from RAM,\n"
            "cold rows pread. Sets EXL2_NGRAM_MODE. Never resident in VRAM."
        ),
    )
    ngram_hybrid_mb: Optional[int] = Field(
        1024,
        description=(
            "Host-RAM budget for the hybrid ngram row cache in MB (default: 1024).\n"
            "Only used when ngram_mode is 'hybrid'. Sets EXL2_NGRAM_HYBRID_MB."
        ),
    )
    priority_preempt: Optional[bool] = Field(
        False,
        description=(
            "Mid-prefill preemption (default: False, opt-in).\n"
            "With priority_queue on, admit newly arrived single-chunk short jobs\n"
            "between chunks of a long prefill so they decode while it continues.\n"
            "The earlier slot-reuse crash chain (negative gather delta, pending-list\n"
            "races, batch MTP 'recommit: invalid position') is fixed: a slot is free\n"
            "only when it holds no job, so a nested admission can never reset the\n"
            "prefilling row. Verified under contention (batch 4, 21k prefill + 5\n"
            "shorts mid-prefill). Without it, short prompts still go ahead of\n"
            "PENDING long prompts and the reprocess penalty applies."
        ),
    )

    _metadata: Metadata = PrivateAttr(Metadata())
    model_config = ConfigDict(protected_namespaces=())


class DraftModelConfig(BaseConfigModel):
    """
    Options for draft models (speculative decoding)
    This will use more VRAM!
    """

    draft_mode: Optional[Literal["model", "disabled", "mtp", "dflash", "ngram"]] = Field(
        "model",
        description=(
            "Drafting mode (default: model).\n"
            "Options: model, disabled, mtp, dflash, ngram.\n"
            "In `model` mode, drafting is disabled if no draft_model_name is provided.\n"
            "ExllamaV2 (ReadyArt fork): `mtp` loads a Gemma-4 assistant drafter\n"
            "(e.g. google/gemma-4-31B-it-assistant) that cross-attends to the target's\n"
            "K/V. Works with the sp and mp tensor-parallel backends.\n"
            "For Qwen3.8-27B / Qwen3.5 (experimental), `mtp` with NO draft_model_name\n"
            "uses the model's built-in MTP drafter (mtp.* weights in the checkpoint).\n"
            "Built-in MTP is greedy-only and single-GPU/autosplit for now.\n"
            "`dflash` loads a DFlash block-diffusion drafter\n"
            "(e.g. z-lab/gemma-4-31B-it-DFlash) that drafts a whole block in one\n"
            "parallel pass, conditioned on the target's hidden states (FlashInfer\n"
            "attention). Requires the mp tensor-parallel backend and max_batch_size 1.\n"
            "A DFlash checkpoint is also auto-detected under `mtp` mode.\n"
            "`ngram` is not supported by the exllamav2 fork paths."
        ),
    )
    # TODO: convert this to a pathlib.path?
    draft_model_dir: Optional[str] = Field(
        "models",
        description="Directory to look for draft models (default: models)",
    )
    draft_model_name: Optional[str] = Field(
        None,
        description=(
            "An initial draft model to load.\nEnsure the model is in the model directory."
        ),
    )
    draft_rope_scale: Optional[float] = Field(
        1.0,
        description=(
            "Rope scale for draft models (default: 1.0).\n"
            "Same as compress_pos_emb.\n"
            "Use if the draft model was trained on long context with rope."
        ),
    )
    draft_rope_alpha: Optional[float] = Field(
        None,
        description=(
            "Rope alpha for draft models (default: None).\n"
            'Same as alpha_value. Set to "auto" to auto-calculate.\n'
            "Leaving this value blank will either pull from the model "
            "or auto-calculate."
        ),
    )
    draft_cache_mode: Optional[CACHE_SIZES] = Field(
        "FP16",
        description=(
            "Cache mode for draft models to save VRAM (default: FP16).\n"
            f"Possible values: {str(CACHE_SIZES)[15:-1]}."
        ),
    )
    draft_gpu_split: List[float] = Field(
        default_factory=list,
        description=(
            "Array of VRAM sizes to split between GPUs, in GB (default: []).\n"
            "If this isn't filled in, the draft model is autosplit."
        ),
    )
    draft_num_tokens: Optional[int] = Field(
        None,
        description=(
            "Number of tokens to draft per iteration (default: draft model default)\n"
            "Recurrent (linear or sliding attention) models use more VRAM for longer drafts.\n"
            "This overhead multiplies with the max batch size, so for models with long drafts\n"
            "(e.g. DFlash with 15 tokens by default) shorter drafts may be preferable."
        ),
    )
    sparse_draft_head: Optional[int] = Field(
        0,
        description=(
            "Draft through a dynamic sparse LM head of this many vocabulary columns;\n"
            "0 uses the full head (default). Cuts draft latency for the MTP and DFlash\n"
            "drafters, which read a full-vocab head per draft step. Verification keeps\n"
            "the full head, so output is unaffected; a probe-based miss gate falls back\n"
            "to the full head if the subset misses too often. The column set follows\n"
            "token frequencies seen in prompts and generations, saved per model under\n"
            "~/.cache/exllamav2-fork/token_stats. Try 16384. ReadyArt exllamav2 fork\n"
            "MP paths only (single-sequence); ignored by the multi-batch scheduler."
        ),
    )
    ngram_match_min: Optional[int] = Field(
        2,
        description=(
            "Minimum match length for exllamav3 n-gram drafting (default: 2).\n"
            "Only used when draft_mode is ngram."
        ),
    )
    draft_device: Optional[int] = Field(
        0,
        description=(
            "Device for the MTP drafter (default: 0).\n"
            "ExllamaV2 ReadyArt fork only. The drafter is a single-device model even\n"
            "when the target runs tensor-parallel."
        ),
        ge=0,
    )
    mtp_draft_graph: Optional[bool] = Field(
        False,
        description=(
            "Capture the whole MTP draft window into one CUDA graph (default: False).\n"
            "ExllamaV2 ReadyArt fork only. Sets EXL2_MTP_DRAFT_GRAPH.\n"
            "FlashInfer paged-decode attention + a GPU-resident embedding table\n"
            "(+2.7 GB on the draft device); draft steps chain on-device with no\n"
            "per-step host round trips. Measured ~68 tok/s generation on\n"
            "Gemma-4-31B-IT (4x3090 200W cap, temp 1, draft_num_tokens 3, ~1.3k ctx).\n"
            "Eager drafting remains the fallback if capture fails."
        ),
    )
    mtp_int8_weights: Optional[bool] = Field(
        False,
        description=(
            "INT8 weight-only MTP drafter GEMVs (default: False). Sets EXL2_MTP_INT8_W.\n"
            "ExllamaV2 ReadyArt fork only. Per-output-row symmetric INT8 + fp16 scales,\n"
            "custom CUDA GEMV kernel (no Triton); halves the drafter's weight-read\n"
            "bandwidth (the m=1 GEMV floor). Draft tokens shift slightly (quantization\n"
            "error) -- speculative acceptance keeps the OUTPUT distribution exact;\n"
            "only the acceptance rate can move. fp16 weights are kept for the sparse\n"
            "head rebuild + fallback."
        ),
    )
    mtp_int8_kv: Optional[bool] = Field(
        False,
        description=(
            "INT8 MTP KV shadow (default: False). Sets EXL2_MTP_INT8_KV.\n"
            "ExllamaV2 ReadyArt fork only. The drafter's gathered FULL-layer target\n"
            "K/V shadow gets an INT8 copy (per-(token,head) scales) read by the int8\n"
            "flash-decode kernel -- halves the long-context cross-attention KV read.\n"
            "SWA rings stay fp16. Adds ~50% of the full-layer shadow VRAM on the draft\n"
            "device. Independent of the target's kv_cache_int8."
        ),
    )


class SamplingConfig(BaseConfigModel):
    """Options for Sampling"""

    override_preset: Optional[str] = Field(
        None,
        description=(
            "Select a sampler override preset (default: None).\n"
            "Find this in the sampler-overrides folder.\n"
            "This overrides default fallbacks for sampler values "
            "that are passed to the API.\n"
            "NOTE: safe_defaults preset provides a fallback for frontends "
            "that do not pass sampling params.\n"
            "Remove it if not necessary."
        ),
    )


class LoraInstanceModel(BaseConfigModel):
    """Model representing an instance of a Lora."""

    name: Optional[str] = None
    scaling: float = Field(1.0, ge=0)


class LoraConfig(BaseConfigModel):
    """Options for Loras"""

    # TODO: convert this to a pathlib.path?
    lora_dir: Optional[str] = Field(
        "loras", description="Directory to look for LoRAs (default: loras)."
    )
    loras: Optional[List[LoraInstanceModel]] = Field(
        None,
        description=(
            "List of LoRAs to load and associated scaling factors "
            "(default scale: 1.0).\n"
            "For the YAML file, add each entry as a YAML list:\n"
            "- name: lora1\n"
            "  scaling: 1.0"
        ),
    )


class EmbeddingsConfig(BaseConfigModel):
    """
    Options for embedding models and loading.
    NOTE: Embeddings requires the "extras" feature to be installed
    Install it via "pip install .[extras]"
    """

    # TODO: convert this to a pathlib.path?
    embedding_model_dir: Optional[str] = Field(
        "models",
        description=("Directory to look for embedding models (default: models)."),
    )
    embeddings_device: Optional[Literal["cpu", "auto", "cuda"]] = Field(
        "cpu",
        description=(
            "Device to load embedding models on (default: cpu).\n"
            "Possible values: cpu, auto, cuda.\n"
            "NOTE: It's recommended to load embedding models on the CPU.\n"
            "If using an AMD GPU, set this value to 'cuda'."
        ),
    )
    embedding_model_name: Optional[str] = Field(
        None,
        description="An initial embedding model to load on the infinity backend.",
    )


class MemoryConfig(BaseConfigModel):
    """Options for development and experimentation"""

    sysmem_recurrent_cache: Optional[int] = Field(
        4096,
        description=("Max size of recurrent cache in system memory, in MB (default: 4096)"),
    )
    cuda_malloc_async: Optional[bool] = Field(
        True,
        description=(
            "Use cudaMallocAsync backend in Torch (default: True).\n"
            "Enabling this is generally preferable, but it may cause issues with certain\n"
            "workloads. Try disabling it if you experience intermittent OoM errors. If\n"
            "False, Torch will use the allocator defined by the system env"
        ),
    )


class DeveloperConfig(BaseConfigModel):
    """Options for development and experimentation"""

    unsafe_launch: Optional[bool] = Field(
        False,
        description=(
            "Skip Exllamav2 version check (default: False).\n"
            "WARNING: It's highly recommended to update your dependencies rather "
            "than enabling this flag."
        ),
    )
    disable_request_streaming: Optional[bool] = Field(
        False, description="Disable API request streaming (default: False)."
    )
    realtime_process_priority: Optional[bool] = Field(
        False,
        description=(
            "Set process to use a higher priority.\n"
            "For realtime process priority, run as administrator or sudo.\n"
            "Otherwise, the priority will be set to high."
        ),
    )
    seqlog: Optional[bool] = Field(
        False,
        description=("Enable extremely verbose seqlog logging, requires a running Seq server"),
    )
    seqlog_server_url: Optional[str] = Field(
        "http://localhost:5341",
        description=("Seq server url:port"),
    )
    seqlog_api_key: Optional[str] = Field(
        None,
        description=("Seq server API key (default: None)"),
    )


class TabbyConfigModel(BaseModel):
    """Base model for a TabbyConfig."""

    config: Optional[ConfigOverrideConfig] = Field(
        default_factory=ConfigOverrideConfig.model_construct,
    )
    network: Optional[NetworkConfig] = Field(
        default_factory=NetworkConfig.model_construct,
    )
    logging: Optional[LoggingConfig] = Field(
        default_factory=LoggingConfig.model_construct,
    )
    model: Optional[ModelConfig] = Field(
        default_factory=ModelConfig.model_construct,
    )
    draft_model: Optional[DraftModelConfig] = Field(
        default_factory=DraftModelConfig.model_construct,
    )
    lora: Optional[LoraConfig] = Field(
        default_factory=LoraConfig.model_construct,
    )
    embeddings: Optional[EmbeddingsConfig] = Field(
        default_factory=EmbeddingsConfig.model_construct,
    )
    sampling: Optional[SamplingConfig] = Field(
        default_factory=SamplingConfig.model_construct,
    )
    memory: Optional[MemoryConfig] = Field(
        default_factory=MemoryConfig.model_construct,
    )
    developer: Optional[DeveloperConfig] = Field(
        default_factory=DeveloperConfig.model_construct,
    )

    model_config = ConfigDict(validate_assignment=True, protected_namespaces=())
