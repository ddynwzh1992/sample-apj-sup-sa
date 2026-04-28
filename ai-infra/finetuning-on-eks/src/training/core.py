"""
Shared training core for SFT fine-tuning.

Contains model loading, LoRA/BnB config, FSDP setup, dataset loading,
trainer creation, and model saving — used by both train.py and train_ray.py.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Unified training configuration."""
    # Model
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    attn_implementation: str = "auto"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Quantization / Distributed
    use_qlora: bool = False
    use_fsdp: bool = False
    use_ddp: str = "auto"  # auto, true, false

    # Training hyperparameters
    batch_size: int = 1
    grad_accum_steps: int = 16
    lr: float = 1e-4
    epochs: int = 1
    max_seq_length: int = 2048

    # Dataset
    dataset_name: str = "tatsu-lab/alpaca"
    max_samples: int = 0

    # Output
    output_dir: str = "./output"

    # Workers (used by train_ray.py for DDP auto-detection)
    num_workers: int = 1


def get_optimal_attention_implementation() -> str:
    """
    Detect GPU architecture and return the optimal attention implementation.

    - sm_80, sm_86, sm_89, sm_90: flash_attention_2 (fastest)
    - sm_120 (Blackwell): sdpa (PyTorch native, fast fallback)
    - Unknown/CPU: eager (safe fallback)
    """
    try:
        if not torch.cuda.is_available():
            logger.info("No CUDA available, using eager attention")
            return "eager"

        major, minor = torch.cuda.get_device_capability()
        compute_cap = f"{major}.{minor}"

        flash_supported = ["8.0", "8.6", "8.9", "9.0"]  # Ampere, Ada, Hopper

        if compute_cap in flash_supported:
            try:
                import flash_attn  # noqa: F401
                logger.info(f"GPU compute capability {compute_cap}: using flash_attention_2")
                return "flash_attention_2"
            except ImportError:
                logger.info("flash_attn not installed, falling back to sdpa")
                return "sdpa"
        elif major >= 12:  # Blackwell (sm_120+)
            logger.info(f"GPU compute capability {compute_cap} (Blackwell): using sdpa")
            return "sdpa"
        else:
            logger.info(f"GPU compute capability {compute_cap}: using eager attention")
            return "eager"
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}, using eager attention")
        return "eager"


def resolve_attention(setting: str) -> str:
    """Resolve 'auto' to detected implementation, or pass through explicit value."""
    if setting == "auto":
        return get_optimal_attention_implementation()
    return setting


def get_bnb_config(use_qlora: bool, dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
    """Return BitsAndBytesConfig for QLoRA, or None."""
    if not use_qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    """Return LoraConfig for CAUSAL_LM."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(
    model_name: str,
    use_qlora: bool = False,
    attn_implementation: str = "flash_attention_2",
    torch_dtype: torch.dtype = torch.bfloat16,
    use_fsdp: bool = False,
    device_map=None,
):
    """
    Load AutoModelForCausalLM and AutoTokenizer.

    Args:
        device_map: For QLoRA in Ray, pass {"": f"cuda:{local_rank}"}.
                    For standalone or FSDP, pass None.
    """
    logger.info(f"Loading model: {model_name}")

    bnb_config = get_bnb_config(use_qlora, torch_dtype)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=model_kwargs["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_qlora:
        gc_for_qlora = not use_fsdp
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc_for_qlora)

    return model, tokenizer


def infer_fsdp_wrap_classes(model) -> List[str]:
    """Detect transformer block class names for FSDP wrapping."""
    COMMON_BLOCKS = {
        "LlamaDecoderLayer", "MistralDecoderLayer", "MixtralDecoderLayer",
        "Qwen2DecoderLayer", "Qwen3MoeDecoderLayer", "Gemma2DecoderLayer",
        "Phi3DecoderLayer", "GPTNeoXLayer", "MPTBlock", "BloomBlock",
        "FalconDecoderLayer", "DecoderLayer", "GPTJBlock", "OPTDecoderLayer",
    }

    hits = set()
    for _, m in model.named_modules():
        if m.__class__.__name__ in COMMON_BLOCKS:
            hits.add(m.__class__.__name__)

    if not hits:
        for _, m in model.named_modules():
            name = m.__class__.__name__
            if any(s in name for s in ["Block", "DecoderLayer", "EncoderLayer", "Layer"]):
                if "Embedding" not in name:
                    hits.add(name)

    return sorted(hits)


def build_fsdp_config(model) -> Tuple[str, dict]:
    """
    Build FSDP configuration for HF Trainer.

    Returns (fsdp_str, fsdp_config_dict) for TrainingArguments.
    """
    fsdp_str = "hybrid_shard auto_wrap"

    fsdp_wrap_classes = infer_fsdp_wrap_classes(model)
    if not fsdp_wrap_classes:
        raise RuntimeError("Could not infer transformer block classes for FSDP wrapping.")

    fsdp_config_dict = {
        "fsdp_transformer_layer_cls_to_wrap": fsdp_wrap_classes,
        "activation_checkpointing": True,
        "activation_checkpointing_reentrant": False,
        "limit_all_gathers": True,
        "use_orig_params": True,
        "sync_module_states": True,
    }

    return fsdp_str, fsdp_config_dict


def load_sft_dataset(name: str, max_samples: int = 0):
    """Load dataset with optional sample limit."""
    dataset = load_dataset(name, split="train")
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def create_sft_trainer(
    model,
    tokenizer,
    dataset,
    lora_config: LoraConfig,
    config: TrainingConfig,
    torch_dtype: torch.dtype,
    gradient_checkpointing: bool,
    fsdp_str: str = "",
    fsdp_config_dict: Optional[dict] = None,
    use_ddp: bool = False,
    local_rank: int = -1,
) -> SFTTrainer:
    """Create SFTTrainer with SFTConfig."""
    training_args = SFTConfig(
        output_dir=config.output_dir,
        max_length=config.max_seq_length,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum_steps,
        learning_rate=config.lr,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        logging_steps=5,
        save_strategy="no",
        fsdp=fsdp_str,
        fsdp_config=fsdp_config_dict,
        report_to="none",
        ddp_find_unused_parameters=True if use_ddp else None,
        local_rank=local_rank if use_ddp else -1,
        packing=False,
        dataloader_num_workers=4,
        dataset_text_field="text",
    )

    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )


def save_model(trainer, tokenizer, output_dir: str, use_fsdp: bool = False):
    """Save model with FSDP-aware handling."""
    if use_fsdp:
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, save_policy):
            trainer.model.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
