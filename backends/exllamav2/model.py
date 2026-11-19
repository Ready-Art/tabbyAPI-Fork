"""The model container class for ExLlamaV2 models."""

import asyncio
import gc
import math
import os
import pathlib
import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2CacheBase,
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
    ExLlamaV2VisionTower,
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2DynamicGeneratorAsync,
    ExLlamaV2DynamicJobAsync,
)
from itertools import zip_longest
from typing import Dict, List, Optional

from backends.base_model_container import BaseModelContainer
from backends.exllamav2.fork_generator import (
    ExLlamaV2ForkGeneratorAsync,
    ExLlamaV2ForkJobAsync,
)
from backends.exllamav2.grammar import (
    ExLlamaV2Grammar,
    clear_grammar_func_cache,
)
from backends.exllamav2.utils import exllama_disabled_flash_attn
from backends.exllamav2.vision import clear_image_embedding_cache
from common.concurrency import iterate_in_threadpool
from common.gen_logging import (
    log_generation_params,
    log_metrics,
    log_prompt,
    log_response,
)
from common.errors import ContextLengthExceededError
from common.logger import xlogger
from common.hardware import hardware_supports_flash_attn
from common.health import HealthManager
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import DisconnectHandler
from common.optional_dependencies import check_package_version
from common.sampling import BaseSamplerRequest
from common.templating import PromptTemplate, find_prompt_template
from common.transformers_utils import HFModel
from common.utils import calculate_rope_alpha, coalesce, unwrap
from endpoints.core.types.model import ModelCard, ModelCardParameters
from endpoints.OAI.utils.tools import is_supported_format

# ReadyArt exllamav2 fork extensions (SWA cache, TP process group, MTP drafter).
# Guarded so the container keeps working with upstream exllamav2 installs.
try:
    from exllamav2.cache import ExLlamaV2Cache_SWA
except ImportError:
    ExLlamaV2Cache_SWA = None

try:
    from exllamav2.mtp_drafter import Gemma4AssistantDrafter
    from exllamav2.tp_mp import TPProcessGroup
except ImportError:
    Gemma4AssistantDrafter = None
    TPProcessGroup = None


class ExllamaV2Container(BaseModelContainer):
    """The model container class for ExLlamaV2 models."""

    # Model directories
    model_dir: pathlib.Path = pathlib.Path("models")
    draft_model_dir: pathlib.Path = pathlib.Path("models")
    prompt_template: Optional[PromptTemplate] = None

    # HF model instance
    hf_model: HFModel

    # Exl2 vars
    config: Optional[ExLlamaV2Config] = None
    model: Optional[ExLlamaV2] = None
    cache: Optional[ExLlamaV2Cache] = None
    tokenizer: Optional[ExLlamaV2Tokenizer] = None
    generator: Optional[ExLlamaV2DynamicGeneratorAsync] = None
    prompt_template: Optional[PromptTemplate] = None
    paged: bool = True

    # Draft model vars
    use_draft_model: bool = False
    draft_config: Optional[ExLlamaV2Config] = None
    draft_model: Optional[ExLlamaV2] = None
    draft_cache: Optional[ExLlamaV2Cache] = None

    # Internal config vars
    cache_size: int = None
    cache_mode: str = "FP16"
    draft_cache_mode: str = "FP16"
    max_batch_size: Optional[int] = None

    # GPU split vars
    gpu_split: List[float] = []
    draft_gpu_split: List[float] = []
    gpu_split_auto: bool = True
    autosplit_reserve: List[float] = [96 * 1024**2]
    use_tp: bool = False

    # ReadyArt fork vars (TP backends, MTP drafting, FlashInfer, CUDA graphs)
    tp_backend: str = "native"
    use_mtp: bool = False
    mtp_drafter: Optional[object] = None
    mtp_window: int = 6
    draft_device: int = 0
    tp_group: Optional[object] = None
    fork_mode: bool = False
    is_swa: bool = False
    is_gemma4: bool = False
    use_flashinfer: bool = True

    # Vision vars
    use_vision: bool = False
    vision_model: Optional[ExLlamaV2VisionTower] = None

    # Load synchronization
    active_job_ids: Dict[str, Optional[ExLlamaV2DynamicJobAsync]] = {}
    loaded: bool = False
    load_lock: asyncio.Lock = asyncio.Lock()
    load_condition: asyncio.Condition = asyncio.Condition()

    @classmethod
    async def create(cls, model_directory: pathlib.Path, hf_model: HFModel, **kwargs):
        """
        Primary asynchronous initializer for model container.

        Kwargs are located in config_sample.yml
        """

        _hf = hf_model.hf_config
        _tok = hf_model.tokenizer_config
        _gen = hf_model.generation_config
        xlogger.debug(
            "Creating ExLlamaV2 model instance",
            {
                "kwargs": kwargs,
                "hf_config": _hf.model_dump(mode="json") if _hf else {},
                "tokenizer_config": _tok.model_dump(mode="json") if _tok else {},
                "generation_config": _gen.model_dump(mode="json") if _gen else {},
            },
        )

        # Create a new instance as a "fake self"
        self = cls()

        # Make sure ExllamaV2 is up to date
        check_package_version("exllamav2", "0.3.1")

        # Initialize config
        self.config = ExLlamaV2Config()
        self.model_dir = model_directory
        self.config.model_dir = str(model_directory.resolve())
        self.hf_model = hf_model

        # Make the max seq len 4096 before preparing the config
        # This is a better default than 2048
        self.config.max_seq_len = 4096

        self.config.prepare()

        # Check if the model arch is compatible with various exl2 features
        self.config.arch_compat_overrides()

        # Sliding-window attention (e.g. Gemma-4) needs the fork's SWA ring cache
        self.is_swa = getattr(self.config, "sliding_window", 0) > 0
        if self.is_swa and ExLlamaV2Cache_SWA is None:
            xlogger.warning(
                "This model uses sliding-window attention, which requires the "
                "ReadyArt exllamav2 fork (ExLlamaV2Cache_SWA). Long-context output "
                "may be incorrect with upstream exllamav2."
            )

        # FlashInfer is a hard requirement of the ReadyArt fork (>= 0.7.4) and is
        # always on; there is no disable switch. Reported on the model card only.
        self.use_flashinfer = True

        # Fork runtime knobs, read by the model/drafter/workers at load time
        os.environ["EXL2_TP_GRAPHS"] = "1" if unwrap(kwargs.get("tp_graphs"), True) else "0"
        os.environ["EXLLAMA_FLASHINFER_PLAN_CACHE"] = (
            "1" if unwrap(kwargs.get("flashinfer_plan_cache"), False) else "0"
        )

        # Set vision state and error if vision isn't supported on the current model
        self.use_vision = unwrap(kwargs.get("vision"), False)
        if self.use_vision and not self.config.vision_model_type:
            xlogger.warning(
                "The provided model does not have vision capabilities that are "
                "supported by ExllamaV2. Vision input is disabled."
            )
            self.use_vision = False

        # Prepare the draft model config if necessary
        draft_args = unwrap(kwargs.get("draft_model"), {})
        draft_model_name = draft_args.get("draft_model_name")
        draft_mode = unwrap(draft_args.get("draft_mode"), "model")

        # MTP drafting (ReadyArt fork): a Gemma-4 assistant drafter that
        # cross-attends to the target's K/V instead of a standalone draft model
        self.use_mtp = draft_mode == "mtp"
        if self.use_mtp:
            if not draft_model_name:
                raise ValueError(
                    "draft_mode 'mtp' requires draft_model_name "
                    "(e.g. google/gemma-4-31B-it-assistant downloaded into the draft model dir)"
                )
            if Gemma4AssistantDrafter is None:
                xlogger.warning(
                    "MTP drafting requires the ReadyArt exllamav2 fork "
                    "(exllamav2.mtp_drafter). Disabling MTP."
                )
                self.use_mtp = False
            else:
                self.draft_model_dir = (
                    pathlib.Path(unwrap(draft_args.get("draft_model_dir"), "models"))
                    / draft_model_name
                )
                self.mtp_window = unwrap(draft_args.get("draft_num_tokens"), 6)
                self.draft_device = unwrap(draft_args.get("draft_device"), 0)
                os.environ["EXL2_MTP_DRAFT_GRAPH"] = (
                    "1" if unwrap(draft_args.get("mtp_draft_graph"), False) else "0"
                )
                xlogger.info(
                    f"MTP drafting enabled: {self.draft_model_dir} "
                    f"(window {self.mtp_window}, device cuda:{self.draft_device})"
                )

        if draft_mode == "ngram":
            xlogger.warning(
                "Config-level ngram drafting is not supported by exllamav2. "
                "Use the speculative_ngram sampler option instead."
            )

        self.use_draft_model = bool(draft_args) and bool(draft_model_name) and draft_mode == "model"

        # Always disable draft if params are incorrectly configured
        if draft_mode == "model" and draft_args and draft_model_name is None:
            xlogger.warning(
                "Draft model is disabled because a model name "
                "wasn't provided. Please check your config.yml!"
            )
            self.use_draft_model = False

        if self.use_draft_model:
            self.draft_config = ExLlamaV2Config()
            draft_model_path = pathlib.Path(unwrap(draft_args.get("draft_model_dir"), "models"))
            draft_model_path = draft_model_path / draft_model_name

            self.draft_gpu_split = unwrap(draft_args.get("draft_gpu_split"), [])
            self.draft_model_dir = draft_model_path
            self.draft_config.model_dir = str(draft_model_path.resolve())
            self.draft_config.prepare()

        # MARK: User configuration

        # Get cache mode
        self.cache_mode = unwrap(kwargs.get("cache_mode"), "FP16")

        # Catch exllamav3 cache_mode
        if self.cache_mode != "FP16" and not self.cache_mode.startswith("Q"):
            xlogger.warning(
                f"Provided cache mode '{self.cache_mode}' is not a "
                "valid choice for exllamav2, please check your settings. "
                "Defaulting to FP16."
            )
            self.cache_mode = "FP16"

        # The SWA ring cache is FP16-only
        if self.is_swa and self.cache_mode != "FP16":
            xlogger.warning(
                "Quantized K/V cache modes are not supported with sliding-window "
                "attention. Defaulting to FP16."
            )
            self.cache_mode = "FP16"

        # Turn off GPU split if the user is using 1 GPU
        gpu_count = torch.cuda.device_count()
        gpu_split_auto = unwrap(kwargs.get("gpu_split_auto"), True)
        use_tp = unwrap(kwargs.get("tensor_parallel"), False)
        gpu_split = unwrap(kwargs.get("gpu_split"), [])
        gpu_device_list = list(range(0, gpu_count))

        # Set GPU split options
        if gpu_count == 1:
            self.gpu_split_auto = False
            xlogger.info("Disabling GPU split because one GPU is in use.")
        else:
            # Set tensor parallel
            if use_tp:
                self.use_tp = True

                # TP has its own autosplit loader
                self.gpu_split_auto = False

            # Enable manual GPU split if provided
            if gpu_split:
                self.gpu_split_auto = False
                self.gpu_split = gpu_split

                gpu_device_list = [
                    device_idx for device_idx, memory in enumerate(self.gpu_split) if memory > 0
                ]
            elif gpu_split_auto and not self.use_tp:
                # Otherwise fallback to autosplit settings
                self.gpu_split_auto = gpu_split_auto

                autosplit_reserve_megabytes = unwrap(kwargs.get("autosplit_reserve"), [96])

                # Reserve VRAM for each GPU
                self.autosplit_reserve = [
                    int(math.ceil(value * 1024**2)) for value in autosplit_reserve_megabytes
                ]

            # Change the GPU device list only if gpu_split's list is too small
            # This allows for an uneven list specification
            if self.draft_gpu_split and len(self.draft_gpu_split) > len(self.gpu_split):
                gpu_device_list = [
                    device_idx
                    for device_idx, memory in enumerate(self.draft_gpu_split)
                    if memory > 0
                ]

        # Tensor-parallel backend (ReadyArt fork: sp = single-process streaming TP,
        # mp = one worker process per GPU; nccl is an alias for mp)
        self.tp_backend = unwrap(kwargs.get("tensor_parallel_backend"), "native")
        if self.tp_backend == "nccl":
            self.tp_backend = "mp"
        if self.tp_backend in ("sp", "mp"):
            if self.tp_backend == "mp" and TPProcessGroup is None:
                xlogger.warning(
                    "The 'mp' tensor-parallel backend requires the ReadyArt exllamav2 "
                    "fork (exllamav2.tp_mp). Falling back to native TP."
                )
                self.tp_backend = "native"
            else:
                self.use_tp = True
                self.gpu_split_auto = False
                xlogger.info(f"Using fork tensor-parallel backend: {self.tp_backend}")
        elif self.tp_backend != "native":
            xlogger.warning(
                f"Unknown tensor_parallel_backend '{self.tp_backend}' for exllamav2. "
                "Options: native, sp, mp. Defaulting to native."
            )
            self.tp_backend = "native"

        # The paged TP attention kernel has no sliding-window support; SWA models
        # must use the fork's streaming (sp) or multiprocess (mp) path
        if self.is_swa and self.use_tp and self.tp_backend == "native":
            if ExLlamaV2Cache_SWA is not None:
                xlogger.warning(
                    "Sliding-window model with tensor parallelism: switching to the "
                    "fork 'sp' backend (the paged TP kernel does not support SWA)."
                )
                self.tp_backend = "sp"
            else:
                xlogger.warning(
                    "Sliding-window model with tensor parallelism requires the "
                    "ReadyArt exllamav2 fork. Falling back to autosplit."
                )
                self.use_tp = False
                self.gpu_split_auto = True

        # MTP drafting rides on the streaming generator, not the dynamic one
        if self.use_mtp and self.use_tp and self.tp_backend == "native":
            xlogger.info("MTP drafting with TP: switching to the fork 'sp' backend.")
            self.tp_backend = "sp"

        self.fork_mode = self.use_mtp or self.tp_backend in ("sp", "mp")

        # SWA models on the non-TP path also use the streaming wrapper: the paged
        # dynamic generator's SWA ring protocol (prefill window_beg/ring shifts,
        # page reuse over rings) is not verified here, while the streaming path's
        # ring handling is. One job at a time, fresh prefill per job.
        if self.is_swa and not self.use_tp and ExLlamaV2Cache_SWA is not None:
            if not self.fork_mode:
                xlogger.info(
                    "Sliding-window model: using the streaming generation path "
                    "(batch size 1) for verified SWA ring handling."
                )
            self.fork_mode = True

        if self.fork_mode and self.use_vision:
            xlogger.warning(
                "Vision input is not supported by the fork TP/MTP paths. Disabling vision."
            )
            self.use_vision = False

        # Hardcode max output length to 16 (MTP verify windows may need more)
        self.config.max_output_len = max(16, self.mtp_window + 1 if self.use_mtp else 16)

        # Set max batch size to the config override
        self.max_batch_size = unwrap(kwargs.get("max_batch_size"))
        if self.fork_mode:
            # The fork TP/MTP paths own a single cache: one job at a time
            if self.max_batch_size is None or self.max_batch_size > 1:
                xlogger.info("Fork TP/MTP mode: limiting max_batch_size to 1.")
            self.max_batch_size = 1

        # Check whether the user's configuration supports flash/paged attention
        # Also check if exl2 has disabled flash attention
        if exllama_disabled_flash_attn(
            self.config.no_flash_attn
        ) or not hardware_supports_flash_attn(gpu_device_list):
            gpu_unsupported_message = (
                "An unsupported GPU is found in this configuration. "
                "Switching to compatibility mode. \n"
                "This disables parallel batching "
                "and features that rely on it (ex. CFG). \n"
                "To disable compatability mode, all GPUs must be ampere "
                "(30 series) or newer. AMD GPUs are not supported."
            )

            xlogger.warning(gpu_unsupported_message)

            self.config.no_flash_attn = True
            if self.draft_config:
                self.draft_config.no_flash_attn = True
            self.paged = False
            self.max_batch_size = 1
            torch.backends.cuda.enable_flash_sdp(False)

        # Grab user-set max seq len
        user_max_seq_len = kwargs.get("max_seq_len")

        # Set k/v cache size
        # cache_size is only relevant when paged mode is enabled
        if self.paged:
            user_cache_size = coalesce(kwargs.get("cache_size"), user_max_seq_len, 4096)
            self.cache_size = self.adjust_cache_size(user_cache_size)
            self.config.max_seq_len = self.adjust_max_seq_len(user_max_seq_len)
        else:
            self.config.max_seq_len = unwrap(
                user_max_seq_len,
                min(hf_model.hf_config.get_max_position_embeddings(), 4096),
            )
            self.cache_size = self.config.max_seq_len

        # Set the rope scale
        self.config.scale_pos_emb = unwrap(kwargs.get("rope_scale"), self.config.scale_pos_emb)

        # Sets rope alpha value.
        # Utilize the model's max_position_embeddings as a base value
        # Automatically calculate if unset or defined as an "auto" literal.
        rope_alpha = unwrap(kwargs.get("rope_alpha"), "auto")
        if rope_alpha == "auto":
            self.config.scale_alpha_value = calculate_rope_alpha(
                hf_model.hf_config.get_max_position_embeddings(), self.config.max_seq_len
            )
        else:
            self.config.scale_alpha_value = rope_alpha

        # Try to set prompt template
        self.prompt_template = await find_prompt_template(
            kwargs.get("prompt_template"), model_directory
        )

        # Tool calling
        self.tool_format = kwargs.get("tool_format")
        if self.tool_format and not is_supported_format(self.tool_format):
            xlogger.warning(f"Unrecognized tool_format in config: {self.tool_format}")
            self.tool_format = None
        if self.tool_format:
            xlogger.info(f"Using tool format: {self.tool_format}")

        # Catch all for template lookup errors
        if self.prompt_template:
            xlogger.info(
                f'Using template "{self.prompt_template.name}" for chat completions.',
                {"raw": self.prompt_template.raw_template},
            )
        else:
            xlogger.warning(
                "Chat completions are disabled because a prompt "
                "template wasn't provided or auto-detected."
            )

        # Make sure chunk size is >= 256, keep near or below max seq len
        user_chunk_size = unwrap(kwargs.get("chunk_size"), 2048)
        chunk_size = sorted((256, user_chunk_size, self.config.max_seq_len))[1]
        chunk_remainder = chunk_size % 256
        if chunk_remainder != 0:
            rounded_chunk_size = int(256 * ((chunk_size - chunk_remainder) / 256 + 1))

            xlogger.warning(
                f"The given chunk size ({chunk_size}) is "
                "not a multiple of 256.\n"
                "Overriding chunk_size with an overestimated value of "
                f"{rounded_chunk_size} tokens."
            )
            chunk_size = rounded_chunk_size
        self.config.max_input_len = chunk_size
        self.config.max_attention_size = chunk_size**2

        # Set user-configured draft model values
        if self.use_draft_model:
            self.draft_config.max_seq_len = self.config.max_seq_len

            self.draft_config.scale_pos_emb = unwrap(draft_args.get("draft_rope_scale"), 1.0)

            # Set draft rope alpha. Follows same behavior as model rope alpha.
            # Use the max_position_embeddings of the model
            draft_rope_alpha = unwrap(draft_args.get("draft_rope_alpha"), "auto")
            if draft_rope_alpha == "auto":
                self.draft_config.scale_alpha_value = calculate_rope_alpha(
                    hf_model.hf_config.get_max_position_embeddings(),
                    self.draft_config.max_seq_len,
                )
            else:
                self.draft_config.scale_alpha_value = draft_rope_alpha

            # Set draft cache mode
            self.draft_cache_mode = unwrap(draft_args.get("draft_cache_mode"), "FP16")

            # Catch exllamav3 draft_cache_mode
            if self.draft_cache_mode != "FP16" and not self.draft_cache_mode.startswith("Q"):
                xlogger.warning(
                    f"Provided draft cache mode '{self.draft_cache_mode}' is not a "
                    "valid choice for exllamav2, please check your settings. "
                    "Defaulting to FP16."
                )
                self.draft_cache_mode = "FP16"

            # Edit the draft config size
            if chunk_size:
                self.draft_config.max_input_len = chunk_size
                self.draft_config.max_attention_size = chunk_size**2

        # Reasoning mode
        self.reasoning = kwargs.get("reasoning", False)
        self.reasoning_start_token = kwargs.get("reasoning_start_token", "<think>")
        self.reasoning_end_token = kwargs.get("reasoning_end_token", "</think>")
        self.reasoning_suppress_header = kwargs.get("reasoning_suppress_header", None)
        self.force_enable_thinking = kwargs.get("force_enable_thinking", False)

        # Gemma-4 conveniences: its reasoning channel is <|channel>thought\n ...
        # <channel|> (enabled via <|think|> in the system turn, which the model's
        # chat template injects for enable_thinking), and its native tool-call
        # syntax is handled by the 'gemma4' tool format.
        _hf_cfg = hf_model.hf_config
        self.is_gemma4 = (_hf_cfg.model_type or "").startswith("gemma4") or any(
            "gemma4" in arch.lower() for arch in (_hf_cfg.architectures or [])
        )
        if self.is_gemma4:
            if (
                self.reasoning
                and self.reasoning_start_token == "<think>"
                and self.reasoning_end_token == "</think>"
            ):
                # The thought channel opens with <|channel>thought\n, but that
                # spans multiple tokens; the stream collector matches tags per
                # chunk. Use the single-token opener and strip the "thought\n"
                # header from the reasoning block instead.
                self.reasoning_start_token = "<|channel>"
                self.reasoning_end_token = "<channel|>"
                if self.reasoning_suppress_header is None:
                    self.reasoning_suppress_header = "thought\n"
                xlogger.info(
                    "Gemma-4 detected: using thought-channel reasoning tags "
                    "(<|channel> / <channel|>, suppress header 'thought\\n')."
                )
            if not self.tool_format:
                self.tool_format = "gemma4"
                xlogger.info("Gemma-4 detected: defaulting tool_format to 'gemma4'.")

        # Return the created instance
        return self

    def adjust_cache_size(self, cache_size):
        # Enforce a multiple of 256 for cache size
        # Overestimate to ensure that the cache isn't below max_seq_len
        cache_remainder = cache_size % 256
        if cache_remainder != 0:
            rounded_cache_size = int(256 * ((cache_size - cache_remainder) / 256 + 1))

            xlogger.warning(
                f"The given cache size ({cache_size}) is "
                "not a multiple of 256.\n"
                "Overriding cache_size with an overestimated value of "
                f"{rounded_cache_size} tokens."
            )

            cache_size = rounded_cache_size

        if self.config.max_seq_len > cache_size:
            xlogger.warning(
                f"The given max_seq_len ({self.config.max_seq_len}) is larger than "
                f"the cache size and will be limited to {cache_size} tokens."
            )
            self.config.max_seq_len = cache_size

        # Warn user if cache size may be inadequate for CFG
        if cache_size < 2 * self.config.max_seq_len:
            xlogger.warning(
                f"The given cache_size ({cache_size}) is less than 2 * max_seq_len "
                "and may be too small for requests using CFG. \n"
                "Ignore this warning if you do not plan on using CFG."
            )

        return cache_size

    def adjust_max_seq_len(self, max_seq_len):
        print(f"User max seq len {max_seq_len}")
        if not max_seq_len:
            default_max_seq_len = min(
                self.hf_model.hf_config.get_max_position_embeddings(), self.cache_size
            )

            xlogger.warning(
                f"max_seq_len is undefined. Overriding to {default_max_seq_len} tokens."
            )
            max_seq_len = default_max_seq_len
        elif max_seq_len > self.cache_size:
            xlogger.warning(
                f"The given max_seq_len ({max_seq_len}) is larger than the cache size "
                f"and will be limited to {self.cache_size} tokens."
            )
            max_seq_len = self.cache_size

        return max_seq_len

    def model_info(self):
        draft_model_card: ModelCard = None
        if self.draft_config:
            draft_model_params = ModelCardParameters(
                max_seq_len=self.draft_config.max_seq_len,
                rope_scale=self.draft_config.scale_pos_emb,
                rope_alpha=self.draft_config.scale_alpha_value,
                cache_mode=self.draft_cache_mode,
            )

            draft_model_card = ModelCard(
                id=self.draft_model_dir.name,
                parameters=draft_model_params,
            )
        elif self.use_mtp:
            draft_model_card = ModelCard(
                id=self.draft_model_dir.name,
                parameters=ModelCardParameters(draft_mode="mtp"),
            )

        model_params = ModelCardParameters(
            max_seq_len=self.config.max_seq_len,
            cache_size=self.cache_size,
            rope_scale=self.config.scale_pos_emb,
            rope_alpha=self.config.scale_alpha_value,
            max_batch_size=self.max_batch_size,
            cache_mode=self.cache_mode,
            chunk_size=self.config.max_input_len,
            use_vision=self.use_vision,
            draft=draft_model_card,
            tensor_parallel=self.use_tp,
            tensor_parallel_backend=self.tp_backend if self.use_tp else None,
            use_mtp=self.use_mtp,
            flashinfer=self.use_flashinfer,
        )

        if self.prompt_template:
            model_params.prompt_template = self.prompt_template.name
            model_params.prompt_template_content = self.prompt_template.raw_template

        model_card = ModelCard(
            id=self.model_dir.name,
            parameters=model_params,
        )

        return model_card

    async def wait_for_jobs(self, skip_wait: bool = False):
        """Polling mechanism to wait for pending generation jobs."""

        if not self.generator:
            return

        # Immediately abort all jobs if asked
        if skip_wait:
            xlogger.warning(
                "Immediately terminating all jobs. Clients will have their requests cancelled.\n"
            )

            for job in self.active_job_ids.values():
                if job:
                    await job.cancel()

        while len(self.active_job_ids) > 0:
            await asyncio.sleep(0.01)

    async def load(self, progress_callback=None):
        """
        Load model

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded.

                Prototype:
                def progress(loaded_modules: int, total_modules: int)
        """

        async for _ in self.load_gen(progress_callback):
            pass

    async def load_gen(self, progress_callback=None, **kwargs):
        """Loads a model and streams progress via a generator."""

        # Indicate that model load has started
        # Do this operation under the load lock's context
        try:
            await self.load_lock.acquire()

            # Wait for existing generation jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            # Streaming gen for model load progress
            model_load_generator = self.load_model_sync(progress_callback)
            async for value in iterate_in_threadpool(model_load_generator):
                yield value

            # Create async generator
            await self.create_generator()

            # Clean up any extra vram usage from torch and cuda
            # (Helps reduce VRAM bottlenecking on Windows)
            gc.collect()
            torch.cuda.empty_cache()

            # Cleanup and update model load state
            self.loaded = True
            xlogger.info("Model successfully loaded.")
        finally:
            self.load_lock.release()

            async with self.load_condition:
                self.load_condition.notify_all()

    @torch.inference_mode()
    def load_model_sync(self, progress_callback=None):
        """
        Synchronous generator for loading.

        Args:
            progress_callback (function, optional): A function to call for each
                module loaded.

                Prototype:
                def progress(loaded_modules: int, total_modules: int)

        Runs under a shared inference mode context.
        """

        # Reset tokenizer namespace vars and create a tokenizer
        ExLlamaV2Tokenizer.unspecial_piece_to_id = {}
        ExLlamaV2Tokenizer.unspecial_id_to_piece = {}
        ExLlamaV2Tokenizer.extended_id_to_piece = {}
        ExLlamaV2Tokenizer.extended_piece_to_id = {}

        self.tokenizer = ExLlamaV2Tokenizer(self.config)

        # Calculate autosplit reserve for all GPUs
        gpu_count = torch.cuda.device_count()
        autosplit_reserve = self.autosplit_reserve + [0] * (gpu_count - len(self.autosplit_reserve))

        # Load draft model if a config is present
        if self.draft_config:
            self.draft_model = ExLlamaV2(self.draft_config)
            xlogger.info("Loading draft model: " + self.draft_config.model_dir)

            # Draft uses the autosplit loader, so create a cache that reflects this
            draft_cache_class = self.get_cache_class(self.draft_cache_mode)

            if self.draft_gpu_split:
                xlogger.info("Loading with a manual GPU split (or a one GPU setup)")

                for value in self.draft_model.load_gen(
                    self.draft_gpu_split,
                    callback_gen=progress_callback,
                ):
                    if value:
                        yield value

                self.draft_cache = self.create_cache(
                    cache_class=draft_cache_class,
                    autosplit=False,
                    use_tp=False,
                    model=self.draft_model,
                )
            else:
                xlogger.info("Loading with autosplit")

                self.draft_cache = self.create_cache(
                    cache_class=draft_cache_class,
                    autosplit=True,
                    use_tp=False,
                    model=self.draft_model,
                )

                for value in self.draft_model.load_autosplit_gen(
                    self.draft_cache,
                    reserve_vram=autosplit_reserve,
                    last_id_only=True,
                    callback_gen=progress_callback,
                ):
                    if value:
                        yield value

            # Test VRAM allocation with a full-length forward pass
            input_ids = torch.zeros((1, self.config.max_input_len), dtype=torch.long)
            self.draft_model.forward(input_ids, cache=self.cache, preprocess_only=True)

        # Load vision tower if it exists
        if self.use_vision:
            self.vision_model = ExLlamaV2VisionTower(self.config)

            for value in self.vision_model.load_gen(callback_gen=progress_callback):
                if value:
                    yield value

        # Get class of the model cache (SWA models need the fork's ring cache)
        cache_class = self.get_cache_class(self.cache_mode)
        if self.is_swa and ExLlamaV2Cache_SWA is not None:
            cache_class = ExLlamaV2Cache_SWA

        if self.tp_backend == "mp" and self.use_tp:
            # Multiprocess TP (ReadyArt fork): one worker process per GPU.
            # The parent only holds the master rank's shard; all forwards are
            # collectives driven through TPProcessGroup.
            xlogger.info("Loading with multiprocess tensor parallel (fork mp backend)")

            gpu_split = self.gpu_split
            if not gpu_split:
                gpu_split = [
                    round(torch.cuda.get_device_properties(i).total_memory / 1024**3 - 1.0, 2)
                    for i in range(torch.cuda.device_count())
                ]
                xlogger.info(f"MP auto GPU split (GB): {gpu_split}")

            self.tp_group = TPProcessGroup(
                self.config.model_dir,
                gpu_split,
                max_seq_len=self.config.max_seq_len,
                cache_tokens=self.cache_size,
                cache_base_name="ExLlamaV2Cache_SWA" if self.is_swa else "ExLlamaV2Cache",
            )
            self.tp_group.start()
            self.model = self.tp_group.model
            self.cache = self.tp_group.cache

            # Workers load synchronously inside start(); report a single step
            if progress_callback:
                yield from progress_callback(0, 1)
                yield from progress_callback(1, 1)
            else:
                yield 0, 1
                yield 1, 1
        else:
            self.model = ExLlamaV2(self.config)
            xlogger.info("Loading model: " + self.config.model_dir)

            # Load model with manual split
            # Entrypoint for single GPU users
            if self.use_tp:
                xlogger.info("Loading with tensor parallel")

                # GPU split must be None if the array is empty
                # Otherwise the TP loader fails
                for value in self.model.load_tp_gen(
                    self.gpu_split or None,
                    callback_gen=progress_callback,
                    expect_cache_base=cache_class,
                    expect_cache_tokens=self.cache_size,
                ):
                    if value:
                        yield value
            elif not self.gpu_split_auto:
                xlogger.info("Loading with a manual GPU split (or a one GPU setup)")

                for value in self.model.load_gen(
                    self.gpu_split,
                    callback_gen=progress_callback,
                ):
                    if value:
                        yield value

            # Create the model cache
            self.cache = self.create_cache(
                cache_class=cache_class,
                autosplit=self.gpu_split_auto,
                use_tp=self.use_tp,
                model=self.model,
            )

            # Load model with autosplit (without TP)
            if self.gpu_split_auto and not self.use_tp:
                xlogger.info("Loading with autosplit")

                for value in self.model.load_autosplit_gen(
                    self.cache,
                    reserve_vram=autosplit_reserve,
                    last_id_only=True,
                    callback_gen=progress_callback,
                ):
                    if value:
                        yield value

            # Test VRAM allocation with a full-length forward pass
            input_ids = torch.zeros((1, self.config.max_input_len), dtype=torch.long)
            self.model.forward(input_ids, cache=self.cache, preprocess_only=True)

        # MTP drafter (ReadyArt fork): single-device model that cross-attends to
        # the target's K/V. Works with single-GPU, SP TP and MP TP targets.
        if self.use_mtp:
            xlogger.info(f"Loading MTP drafter: {self.draft_model_dir}")
            try:
                # MP: the drafter's target must be the process group itself so
                # K/V gathers go through gather_kv_mtp (worker pipes).
                drafter_target = self.tp_group if self.tp_group is not None else self.model
                self.mtp_drafter = Gemma4AssistantDrafter(
                    drafter_target,
                    str(self.draft_model_dir.resolve()),
                    device=self.draft_device,
                    num_draft_tokens=self.mtp_window,
                )
                if self.tp_group is not None:
                    self.tp_group.enable_mtp(self.mtp_drafter, num_draft_tokens=self.mtp_window)
            except Exception as ex:
                xlogger.warning(
                    f"Failed to initialize the MTP drafter ({ex}). Disabling MTP."
                )
                self.mtp_drafter = None
                self.use_mtp = False
                self.fork_mode = self.tp_backend in ("sp", "mp")

    # TODO: Maybe make a wrapper class with an ID instead of a utility function
    def get_cache_class(self, cache_mode: str):
        """Utility function to get a cache class based on user preference."""

        match cache_mode:
            case "Q4":
                return ExLlamaV2Cache_Q4
            case "Q6":
                return ExLlamaV2Cache_Q6
            case "Q8":
                return ExLlamaV2Cache_Q8
            case _:
                return ExLlamaV2Cache

    def create_cache(
        self,
        cache_class: ExLlamaV2CacheBase,
        autosplit: bool,
        use_tp: bool,
        model: ExLlamaV2,
    ):
        """Utility function to create a model cache."""

        if use_tp:
            return ExLlamaV2Cache_TP(
                model,
                base=cache_class,
                max_seq_len=self.cache_size,
                batch_size=1,
            )
        else:
            return cache_class(
                model,
                max_seq_len=self.cache_size,
                lazy=autosplit,
                batch_size=1,
            )

    async def create_generator(self):
        """Create and save a Exllama generator class."""

        try:
            # Don't acquire locks unless a model is loaded
            if self.loaded:
                await self.load_lock.acquire()

                # Immediately cancel all jobs
                await self.wait_for_jobs(skip_wait=True)

            # Create new generator
            if self.fork_mode:
                # Fork TP/MTP paths: streaming-generator-based async wrapper
                # (single job at a time), see backends/exllamav2/fork_generator.py
                self.generator = ExLlamaV2ForkGeneratorAsync(
                    model=self.model,
                    cache=self.cache,
                    tokenizer=self.tokenizer,
                    tp_group=self.tp_group,
                    mtp_drafter=self.mtp_drafter,
                    mtp_window=self.mtp_window,
                )
            else:
                self.generator = ExLlamaV2DynamicGeneratorAsync(
                    model=self.model,
                    cache=self.cache,
                    draft_model=self.draft_model,
                    draft_cache=self.draft_cache,
                    tokenizer=self.tokenizer,
                    max_batch_size=self.max_batch_size,
                    paged=self.paged,
                )

            # Update the state of the container var
            if self.max_batch_size is None:
                self.max_batch_size = self.generator.generator.max_batch_size
        finally:
            # This means the generator is being recreated
            # The load lock is already released in the load function
            if self.loaded:
                self.load_lock.release()

                async with self.load_condition:
                    self.load_condition.notify_all()

    def get_loras(self):
        """Convenience function to get all loras."""

        return unwrap(self.generator.generator.current_loras, [])

    async def load_loras(self, lora_directory: pathlib.Path, **kwargs):
        """Load loras."""

        loras = unwrap(kwargs.get("loras"), [])

        if self.tp_backend == "mp" and loras:
            xlogger.warning(
                "LoRAs are not supported with the MP tensor-parallel backend "
                "(workers load sharded weights without LoRA support)."
            )
            return {
                "success": [],
                "failure": [lora.get("name") for lora in loras],
            }

        try:
            await self.load_lock.acquire()

            # Wait for existing generation jobs to finish
            await self.wait_for_jobs(kwargs.get("skip_wait"))

            loras_to_load: List[ExLlamaV2Lora] = []
            success: List[str] = []
            failure: List[str] = []

            for lora in loras:
                lora_name = lora.get("name")
                lora_scaling = unwrap(lora.get("scaling"), 1.0)

                if lora_name is None:
                    xlogger.warning(
                        "One of your loras does not have a name. Please check your "
                        "config.yml! Skipping lora load."
                    )
                    failure.append(lora_name)
                    continue

                xlogger.info(f"Adding lora: {lora_name} at scaling {lora_scaling}")
                lora_path = lora_directory / lora_name

                loras_to_load.append(
                    ExLlamaV2Lora.from_directory(self.model, lora_path, lora_scaling)
                )
                xlogger.info(f"Lora successfully added: {lora_name}")
                success.append(lora_name)

            self.generator.generator.set_loras(loras_to_load)
            xlogger.info("All loras successfully loaded")

            # Return success and failure names
            return {"success": success, "failure": failure}
        finally:
            self.load_lock.release()

            async with self.load_condition:
                self.load_condition.notify_all()

    async def unload(self, loras_only: bool = False, **kwargs):
        """Free all VRAM resources used by the model (and loras)."""

        # Shutdown immediately unloads and bypasses all locks
        do_shutdown = kwargs.get("shutdown")

        try:
            if not do_shutdown:
                await self.load_lock.acquire()

                # Wait for other jobs to finish
                await self.wait_for_jobs(kwargs.get("skip_wait"))

            # Delete references held in the grammar module
            clear_grammar_func_cache()

            # Clear the image embedding cache
            clear_image_embedding_cache()

            # Unload LoRAs
            if self.generator and self.generator.generator.current_loras:
                for lora in self.generator.generator.current_loras:
                    lora.unload()

                self.generator.generator.set_loras([])

            # Unload the entire model if not just unloading loras
            if not loras_only:
                # Shut down MP worker processes before freeing the master shard
                if self.tp_group is not None:
                    self.tp_group.close()
                    self.tp_group = None

                if self.model:
                    self.model.unload()
                self.model = None

                self.mtp_drafter = None

                if self.vision_model:
                    self.vision_model.unload()

                self.vision_model = None

                if self.draft_model:
                    self.draft_model.unload()
                self.draft_model = None

                self.config = None
                self.cache = None
                self.tokenizer = None

                # Cleanup the generator from any pending jobs
                if self.generator is not None:
                    await self.generator.close()
                    self.generator = None

                # Set all model state variables to False
                self.loaded = False

            gc.collect()
            torch.cuda.empty_cache()

            xlogger.info("Loras unloaded." if loras_only else "Model unloaded.")
        finally:
            if not do_shutdown:
                self.load_lock.release()

                async with self.load_condition:
                    self.load_condition.notify_all()

    def encode_tokens(self, text: str, **kwargs):
        """Wrapper to encode tokens from a text string."""

        mm_embeddings: MultimodalEmbeddingWrapper = kwargs.get("embeddings")
        mm_embeddings_content = mm_embeddings.content if mm_embeddings else []

        return (
            self.tokenizer.encode(
                text,
                add_bos=unwrap(kwargs.get("add_bos_token"), self.hf_model.add_bos_token()),
                encode_special_tokens=unwrap(kwargs.get("encode_special_tokens"), True),
                embeddings=mm_embeddings_content,
            )
            .flatten()
            .tolist()
        )

    def decode_tokens(self, ids: List[int], **kwargs):
        """Wrapper to decode tokens from a list of IDs"""

        ids = torch.tensor([ids])
        return self.tokenizer.decode(
            ids,
            decode_special_tokens=unwrap(kwargs.get("decode_special_tokens"), True),
        )[0]

    def get_special_tokens(self):
        return {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
        }

    def validate_context_length(
        self,
        prompt: str,
        params: BaseSamplerRequest,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ):
        prompts = [prompt]
        if params.cfg_scale not in [None, 1.0] and self.paged:
            prompts.append(unwrap(params.negative_prompt, self.tokenizer.bos_token))

        context_lengths = [
            len(
                self.encode_tokens(
                    current_prompt,
                    add_bos_token=unwrap(params.add_bos_token, self.hf_model.add_bos_token()),
                    embeddings=mm_embeddings,
                )
            )
            for current_prompt in prompts
        ]
        context_len = max(context_lengths)
        if context_len > self.config.max_seq_len:
            preamble = "Negative prompt" if context_lengths.index(context_len) == 1 else "Prompt"
            raise ContextLengthExceededError(
                f"{preamble} length {context_len} exceeds the available context size "
                f"of {self.config.max_seq_len} tokens"
            )

    def get_logprobs(self, token_ids: torch.Tensor, token_probs: torch.Tensor):
        top_tokens = [
            self.tokenizer.extended_id_to_piece.get(
                index, self.tokenizer.get_id_to_piece_list(True)[index]
            )
            for index in token_ids.flatten().tolist()
        ]

        top_values = torch.log(token_probs).flatten().tolist()

        # Cannot return -inf in JSON
        cleaned_values = [-1000 if value == float("-inf") else value for value in top_values]

        return dict(zip_longest(top_tokens, cleaned_values))

    async def generate(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        disconnect_handler: DisconnectHandler = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ):
        """Generate a response to a prompt."""
        generations = []
        async for generation in self.stream_generate(
            request_id,
            prompt,
            params,
            disconnect_handler,
            mm_embeddings,
        ):
            if generation is None:
                continue
            generations.append(generation)

        joined_generation = {
            "request_id": "",
            "text": "",
            "full_text": "",
            "prompt_tokens": 0,
            "gen_tokens": 0,
            "tool_calls": None,
            "offset": [],
            "token_probs": {},
            "logprobs": [],
        }

        if generations:
            # Get finish_reason first and then shift where -1 points to
            if "finish_reason" in generations[-1]:
                finish_chunk = generations.pop()
                joined_generation = {**joined_generation, **finish_chunk}
                joined_generation["text"] = joined_generation.get("full_text", "")
            else:
                joined_generation["finish_reason"] = "stop"

        if len(generations) > 0:
            for generation in generations:
                joined_generation["offset"].append(unwrap(generation.get("offset"), -1))
                joined_generation["token_probs"].update(unwrap(generation.get("token_probs"), {}))

                # Include empty logprob dicts for index preservation
                joined_generation["logprobs"].append(unwrap(generation.get("logprobs"), {}))

            joined_generation["prompt_tokens"] = unwrap(generations[-1].get("prompt_tokens"), 0)
            joined_generation["generated_tokens"] = unwrap(
                generations[-1].get("generated_tokens"), 0
            )

        return joined_generation

    async def stream_generate(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        disconnect_handler: DisconnectHandler = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
        filter_trigger: str = None,
    ):
        try:
            # Wait for load lock to be freed before processing
            # Mainly used for loras and other operations where the class is available
            async with self.load_condition:
                await self.load_condition.wait_for(lambda: not self.load_lock.locked())

            # If the model is being unloaded, don't accept new requests
            if not self.loaded:
                raise RuntimeError(
                    "Model is being unloaded. Cannot process new generation requests."
                )

            # Mark that the job is running
            self.active_job_ids[request_id] = None

            # Yield from the internal generator
            async for generation_chunk in self.generate_gen(
                request_id=request_id,
                prompt=prompt,
                params=params,
                disconnect_handler=disconnect_handler,
                mm_embeddings=mm_embeddings,
            ):
                yield generation_chunk
        finally:
            # Clean up and remove the job from active IDs
            del self.active_job_ids[request_id]

    def check_unsupported_settings(self, params: BaseSamplerRequest):
        """
        Check and warn the user if a sampler is unsupported.

        Meant for dev wheels!
        """

        return params

    def assign_gen_params(
        self,
        params: BaseSamplerRequest,
        gen_settings: ExLlamaV2Sampler.Settings,
        grammar_handler: ExLlamaV2Grammar,
    ):
        # Apply settings
        gen_settings.temperature = params.temperature
        gen_settings.temperature_last = params.temperature_last
        gen_settings.smoothing_factor = params.smoothing_factor
        gen_settings.top_k = params.top_k
        gen_settings.top_p = params.top_p
        gen_settings.top_a = params.top_a
        gen_settings.min_p = params.min_p
        gen_settings.tfs = params.tfs
        gen_settings.typical = params.typical
        gen_settings.mirostat = params.mirostat_mode == 2
        gen_settings.skew = params.skew

        # XTC
        if params.xtc_probability > 0.0:
            gen_settings.xtc_probability = params.xtc_probability

            # 0.1 is the default for this value
            gen_settings.xtc_threshold = params.xtc_threshold

        # DynaTemp settings
        max_temp = params.max_temp
        min_temp = params.min_temp

        if params.max_temp > params.min_temp:
            gen_settings.max_temp = max_temp
            gen_settings.min_temp = min_temp
            gen_settings.temp_exponent = params.temp_exponent
        else:
            # Force to default values
            gen_settings.max_temp = 1.0
            gen_settings.min_temp = 1.0
            gen_settings.temp_exponent = 1.0

        # Warn if max/min temp values are > 0
        # and if they're less than or equal to each other
        if max_temp < min_temp or (1 not in {min_temp, max_temp} and max_temp == min_temp):
            xlogger.warning("Max temp is less than or equal to min temp, skipping DynaTemp.")

        # Default tau and eta fallbacks don't matter if mirostat is off
        gen_settings.mirostat_tau = params.mirostat_tau
        gen_settings.mirostat_eta = params.mirostat_eta

        # Penalties
        gen_settings.token_repetition_penalty = params.repetition_penalty
        gen_settings.token_frequency_penalty = params.frequency_penalty
        gen_settings.token_presence_penalty = params.presence_penalty

        # Applies for all penalties despite being called token_repetition_range
        gen_settings.token_repetition_range = unwrap(params.penalty_range, self.config.max_seq_len)

        # Always make sure the fallback is 0 if range < 0
        # It's technically fine to use -1, but this just validates the passed
        # fallback
        # Always default to 0 if something goes wrong
        if gen_settings.token_repetition_range < 0:
            fallback_decay = 0
        else:
            fallback_decay = gen_settings.token_repetition_range
        gen_settings.token_repetition_decay = coalesce(params.repetition_decay, fallback_decay, 0)

        # DRY options
        dry_multiplier = params.dry_multiplier

        # < 0 = disabled
        if dry_multiplier > 0:
            gen_settings.dry_multiplier = dry_multiplier
            gen_settings.dry_allowed_length = params.dry_allowed_length
            gen_settings.dry_base = params.dry_base

            # Exl2 has dry_range as 0 for unlimited unlike -1 for penalty_range
            # Use max_seq_len as the fallback to stay consistent
            gen_settings.dry_range = unwrap(params.dry_range, self.config.max_seq_len)

            # Tokenize sequence breakers
            if params.dry_sequence_breakers:
                gen_settings.dry_sequence_breakers = {
                    self.encode_tokens(s)[-1] for s in params.dry_sequence_breakers
                }

        # Add JSON schema filter if it exists
        if params.json_schema:
            grammar_handler.add_json_schema_filter(params.json_schema, self.model, self.tokenizer)

        # Add regex filter if it exists
        if params.regex_pattern:
            grammar_handler.add_regex_filter(params.regex_pattern, self.model, self.tokenizer)

        # Add EBNF filter if it exists
        if params.grammar_string:
            grammar_handler.add_kbnf_filter(params.grammar_string, self.model, self.tokenizer)

        # Speculative Ngram
        if self.use_mtp and params.speculative_ngram:
            xlogger.warning(
                "speculative_ngram is ignored while MTP drafting is active."
            )
        self.generator.speculative_ngram = params.speculative_ngram

        # Override sampler settings for temp = 0
        if gen_settings.temperature == 0:
            gen_settings.temperature = 1.0
            gen_settings.top_k = 1
            gen_settings.top_p = 0
            gen_settings.typical = 0

            xlogger.warning(
                "Temperature is set to 0. Overriding temp, "
                "top_k, top_p, and typical to 1.0, 1, 0, and 0."
            )

        # Set banned tokens
        if params.banned_tokens:
            gen_settings.disallow_tokens(self.tokenizer, params.banned_tokens)

        # Set allowed tokens
        if params.allowed_tokens:
            gen_settings.allow_tokens(self.tokenizer, params.allowed_tokens)

        # Set logit bias
        if params.logit_bias:
            # Create a vocab tensor if it doesn't exist for token biasing
            if gen_settings.token_bias is None:
                padding = -self.tokenizer.config.vocab_size % 32
                gen_settings.token_bias = torch.zeros(
                    (self.tokenizer.config.vocab_size + padding,),
                    dtype=torch.float,
                )

            # Map logits to the tensor with their biases
            for token_id, bias in params.logit_bias.items():
                if 0 <= token_id < len(self.tokenizer.get_id_to_piece_list(True)):
                    gen_settings.token_bias[token_id] = bias
                else:
                    xlogger.warning(
                        f"Logit bias: Token {token_id} not present in the model's vocab. Skipping."
                    )

    # Adds logprobs to a generation chunk
    def handle_logprobs(self, result: dict, generation: dict):
        top_tokens = unwrap(
            result.get("top_k_tokens"),
            torch.empty((1, 0, 1), dtype=torch.long),
        )

        top_probs = unwrap(
            result.get("top_k_probs"),
            torch.empty((1, 0, 1), dtype=torch.float),
        )

        if top_tokens.numel() > 0 and top_probs.numel() > 0:
            logprobs = self.get_logprobs(top_tokens, top_probs)
            generation["logprobs"] = logprobs

            # The first logprob is the selected token prob
            generation["token_probs"] = {
                token: logprobs[token] for token in list(logprobs.keys())[:1]
            }

    # Creates and returns a finish chunk
    def handle_finish_chunk(self, result: dict, request_id: str, full_text: str):
        eos_reason = result.get("eos_reason")

        stop_str = None
        if eos_reason == "max_new_tokens":
            finish_reason = "length"
        else:
            finish_reason = "stop"
            # Grab stop string if stop was the reason
            if eos_reason == "stop_token":
                stop_str = result.get("eos_triggering_token_str")
            elif eos_reason == "stop_string":
                stop_str = result.get("eos_triggering_string")

        # Prompt
        prompt_tokens = result.get("prompt_tokens")
        cached_tokens = round(result.get("cached_tokens"), 2)
        prompt_time = round(result.get("time_prefill"), 2)
        prompt_ts = (
            "Indeterminate"
            if prompt_time == 0
            else round((prompt_tokens - cached_tokens) / prompt_time, 2)
        )

        # Generated
        gen_tokens = result.get("new_tokens")
        gen_time = result.get("time_generate")
        gen_ts = "Indeterminate" if gen_time == 0 else round(gen_tokens / gen_time, 2)

        # Queue + Total
        queue_time = result.get("time_enqueued")
        total_time = round(queue_time + prompt_time + gen_time, 2)

        finish_chunk = {
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "prompt_time": round(prompt_time, 2),
            "prompt_tokens_per_sec": prompt_ts,
            "gen_tokens": gen_tokens,
            "gen_time": round(gen_time, 2),
            "gen_tokens_per_sec": gen_ts,
            "total_time": total_time,
            "queue_time": round(queue_time, 2),
            "cached_tokens": cached_tokens,
            "finish_reason": finish_reason,
            "stop_str": stop_str,
            "full_text": full_text,
        }

        return finish_chunk

    async def generate_gen(
        self,
        request_id: str,
        prompt: str,
        params: BaseSamplerRequest,
        disconnect_handler: DisconnectHandler = None,
        mm_embeddings: Optional[MultimodalEmbeddingWrapper] = None,
    ):
        """
        Create generator function for prompt completion.

        for kwargs, check common/sampling.py
        """

        prompts = [prompt]
        gen_settings = ExLlamaV2Sampler.Settings()
        grammar_handler = ExLlamaV2Grammar()

        self.assign_gen_params(
            params,
            gen_settings,
            grammar_handler,
        )

        # Set banned strings
        banned_strings = params.banned_strings
        if banned_strings and len(grammar_handler.filters) > 0:
            xlogger.warning(
                "Disabling banned_strings because they cannot be used with grammar filters."
            )

            banned_strings = []

        # Set CFG scale and negative prompt
        cfg_scale = params.cfg_scale
        negative_prompt = None
        if cfg_scale not in [None, 1.0]:
            if self.fork_mode:
                xlogger.warning(
                    "CFG is not supported by the fork TP/MTP paths. Ignoring cfg_scale."
                )
            elif self.paged:
                gen_settings.cfg_scale = cfg_scale

                # If the negative prompt is empty, use the BOS token
                negative_prompt = unwrap(params.negative_prompt, self.tokenizer.bos_token)

                prompts.append(negative_prompt)
            else:
                xlogger.warning(
                    "CFG is currently disabled because paged mode is disabled. "
                    "Please use an ampere (30 series) or higher GPU for CFG support."
                )

        # Dynamically scale penalty range to output tokens
        # Only do this if freq/pres pen is enabled
        # and the repetition range is -1
        auto_scale_penalty_range = (
            gen_settings.token_frequency_penalty != 0 or gen_settings.token_presence_penalty != 0
        ) and gen_settings.token_repetition_range == -1

        stop_conditions = params.stop
        ban_eos_token = params.ban_eos_token

        # Set add_bos_token for generation
        add_bos_token = unwrap(params.add_bos_token, self.hf_model.add_bos_token())

        # Fetch EOS tokens from the HF model if they exist
        eos_tokens = self.hf_model.eos_tokens() or [self.tokenizer.eos_token_id]

        # Ban the EOS token if specified. If not, append to stop conditions
        # as well.
        # Set this below logging to avoid polluting the stop strings array
        if ban_eos_token:
            gen_settings.disallow_tokens(self.tokenizer, eos_tokens)
        else:
            stop_conditions += eos_tokens

        # Gemma-4: also stop when the model opens a new turn (<|turn>). The
        # end-of-turn token <turn|> and the <|tool_response> guard are already
        # part of the HF EOS list. Do NOT stop on <channel|>: the reasoning
        # parser must see the closing thought tag to switch back to content.
        if self.is_gemma4 and not ban_eos_token:
            turn_start = self.tokenizer.single_id("<|turn>")
            if turn_start is not None and turn_start >= 0:
                stop_conditions = list(stop_conditions) + [turn_start]

        # Get multimodal embeddings if present
        mm_embeddings_content = mm_embeddings.content if mm_embeddings else []

        # Encode both positive and negative prompts
        input_ids = [
            self.tokenizer.encode(
                prompt,
                add_bos=add_bos_token,
                encode_special_tokens=True,
                embeddings=mm_embeddings_content,
            )
            for prompt in prompts
        ]

        # The first index will always be the positive prompt
        context_len = input_ids[0].size(dim=-1)

        # The second index will be the negative prompt if CFG is enabled
        negative_context_len = input_ids[1].size(dim=-1) if negative_prompt else 0

        # Automatically set max_tokens to fill up the context
        max_tokens = unwrap(params.max_tokens, 0)
        if max_tokens <= 0:
            max_tokens = self.config.max_seq_len - max(context_len, negative_context_len)

        # Determine if the negative context or the context length is bigger
        context_to_check = max(negative_context_len, context_len)

        # Check total length of prompt against max context length
        if context_to_check > self.config.max_seq_len:
            preamble = "Negative prompt" if negative_context_len > context_len else "Prompt"

            raise ContextLengthExceededError(
                f"{preamble} length {context_to_check} exceeds the available context size "
                f"of {self.config.max_seq_len} tokens"
            )

        # Check total required pages for CFG request to avoid overallocation
        if negative_prompt and (
            sum(
                256 * math.ceil((context + max_tokens) / 256)
                for context in (context_len, negative_context_len)
            )
            > self.cache_size
        ):
            raise ValueError(
                f"Total required page size for request "
                f"{context_len} + {negative_context_len} + {max_tokens} * 2 "
                f"is greater than cache_size {self.cache_size}"
            )

        # Log prompt to console. Add the BOS token if specified
        log_prompt(
            f"{self.tokenizer.bos_token if add_bos_token else ''}{prompt}",
            request_id,
            negative_prompt,
        )

        # Create and add a new job
        # Don't use the request ID here as there can be multiple jobs per request
        if self.fork_mode:
            if mm_embeddings_content:
                raise ValueError(
                    "Vision input is not supported by the fork TP/MTP paths"
                )
            job_cls = ExLlamaV2ForkJobAsync
        else:
            job_cls = ExLlamaV2DynamicJobAsync

        job = job_cls(
            self.generator,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            min_new_tokens=params.min_tokens,
            gen_settings=gen_settings,
            stop_conditions=stop_conditions,
            decode_special_tokens=True,
            filters=grammar_handler.filters,
            filter_prefer_eos=bool(grammar_handler.filters),
            return_probs=params.logprobs > 0,
            return_top_tokens=params.logprobs,
            return_logits=params.logprobs > 0,
            banned_strings=banned_strings,
            token_healing=params.token_healing,
            identifier=request_id,
            embeddings=mm_embeddings_content,
        )
        await disconnect_handler.add_cleanup_task(id(job), job.cancel, ())

        # Assign the active job to the request ID
        self.active_job_ids[request_id] = job

        # Save generated tokens and full response
        # Copy over max seq len incase model is unloaded and stored jobs can complete
        # Full response is required for offset calculation
        max_seq_len = self.config.max_seq_len
        generated_tokens = 0
        full_response = ""
        metrics_result = {}

        # Get the generation status once it's ready
        try:
            async for result in job:
                # Abort if the event is set while streaming
                await disconnect_handler.poll()

                stage = result.get("stage")
                result_id = result.get("identifier")

                if stage == "streaming" and result_id == request_id:
                    chunk = unwrap(result.get("text"), "")
                    full_response += chunk

                    chunk_tokens = result.get("token_ids")
                    if chunk_tokens is not None:
                        generated_tokens += chunk_tokens.size(dim=0)

                    generation = {
                        "request_id": request_id,
                        "text": chunk,
                        "prompt_tokens": context_len,
                        "generated_tokens": generated_tokens,
                        "offset": len(full_response),
                    }

                    # Increase penalty range to generated token amount
                    if auto_scale_penalty_range:
                        gen_settings.token_repetition_range = generated_tokens

                    # Handle logprobs
                    if params.logprobs > 0:
                        self.handle_logprobs(result, generation)

                    yield generation

                    # Yield a finish chunk when generation is finished
                    if result.get("eos"):
                        log_response(request_id, full_response)
                        finish_chunk = self.handle_finish_chunk(result, request_id, full_response)
                        await disconnect_handler.finish(id(job))

                        # Save the final result for metrics logging
                        metrics_result = finish_chunk

                        yield finish_chunk
                        break
        except asyncio.CancelledError:
            if not job.cancelled:
                await job.cancel()

        except Exception as ex:
            # Create a new generator since the current state is broken
            # No need to wait for this to finish
            xlogger.error(
                "FATAL ERROR with generation. "
                "Attempting to recreate the generator. "
                "If this fails, please restart the server.\n",
                {"exception": str(ex)},
            )
            asyncio.ensure_future(self.create_generator())

            await HealthManager.add_unhealthy_event(ex)

            raise ex
        finally:
            # Log generation options to console
            # Some options are too large, so log the args instead
            log_generation_params(
                request_id=request_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=eos_tokens,
                prompt=prompt,
                **params.model_dump(exclude={"prompt"}),
                auto_scale_penalty_range=auto_scale_penalty_range,
            )

            # Log the metrics if present
            if metrics_result:
                log_metrics(
                    request_id,
                    metrics_result,
                    context_len,
                    max_seq_len,
                )
