"""
Ray Train wrapper for distributed SFT training.

Uses Ray Train's TorchTrainer for multi-node PyTorch training with FSDP.
This module wraps the core training logic from core.py to enable
distributed training across multiple nodes using Ray.

Features:
    - Distributed checkpointing to S3 for fault tolerance
    - Automatic resume from last checkpoint
    - Configurable checkpoint frequency

Usage:
    # Via RayJob (recommended)
    kubectl apply -f kubernetes/base/ray/rayjob.yaml

    # Direct execution (for testing)
    python train_ray.py

Environment Variables:
    MODEL_NAME: HuggingFace model name (default: meta-llama/Llama-2-7b-hf)
    DATASET_NAME: HuggingFace dataset name (default: tatsu-lab/alpaca)
    USE_QLORA: Enable 4-bit quantization (default: false)
    USE_FSDP: Enable FSDP sharding (default: true for multi-node)
    USE_DDP: Enable DDP - auto/true/false (default: auto)
              auto = enabled when num_workers > 1 and FSDP disabled
    NUM_WORKERS: Number of Ray workers (default: 1)
    GPUS_PER_WORKER: GPUs per worker (default: 1)
    BATCH_SIZE: Per-device batch size (default: 1)
    GRADIENT_ACCUMULATION_STEPS: Gradient accumulation (default: 16)
    LEARNING_RATE: Learning rate (default: 1e-4)
    NUM_EPOCHS: Number of training epochs (default: 1)
    MAX_SEQ_LENGTH: Maximum sequence length (default: 2048)
    LORA_R: LoRA rank (default: 16)
    OUTPUT_DIR: Local output directory fallback (default: /data/outputs)

    S3 Storage (optional - falls back to local if not set):
    RAY_STORAGE_PATH: S3 path for Ray checkpoints (e.g., s3://bucket/ray)
    OUTPUT_PATH: S3 path for model outputs (e.g., s3://bucket/outputs)
    CHECKPOINT_FREQUENCY: Steps between checkpoints (default: 500)
    NUM_CHECKPOINTS_TO_KEEP: Max checkpoints to retain (default: 2)
"""
import os
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any

import ray
from ray import train
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, FailureConfig, Checkpoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_training_config() -> Dict[str, Any]:
    """Read training configuration from environment variables."""
    return {
        "model_name": os.environ.get("MODEL_NAME", "meta-llama/Llama-2-7b-hf"),
        "dataset_name": os.environ.get("DATASET_NAME", "tatsu-lab/alpaca"),
        "use_qlora": os.environ.get("USE_QLORA", "false").lower() == "true",
        "use_fsdp": os.environ.get("USE_FSDP", "true").lower() == "true",
        "batch_size": int(os.environ.get("BATCH_SIZE", "1")),
        "grad_accum_steps": int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "16")),
        "lr": float(os.environ.get("LEARNING_RATE", "1e-4")),
        "epochs": int(os.environ.get("NUM_EPOCHS", "1")),
        "max_seq_length": int(os.environ.get("MAX_SEQ_LENGTH", "2048")),
        "lora_r": int(os.environ.get("LORA_R", "16")),
        "lora_alpha": int(os.environ.get("LORA_ALPHA", "32")),
        "lora_dropout": float(os.environ.get("LORA_DROPOUT", "0.05")),
        "output_dir": os.environ.get("OUTPUT_DIR", "/data/outputs"),
        "max_samples": int(os.environ.get("MAX_SAMPLES", "0")),
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "auto"),
        # Distributed training
        "num_workers": int(os.environ.get("NUM_WORKERS", "1")),
        "gpus_per_worker": int(os.environ.get("GPUS_PER_WORKER", "1")),
        "cpus_per_worker": int(os.environ.get("CPUS_PER_WORKER", "8")),
        "use_ddp": os.environ.get("USE_DDP", "auto").lower(),
        # S3 storage configuration
        "ray_storage_path": os.environ.get("RAY_STORAGE_PATH", "/data/ray_results"),
        "output_path": os.environ.get("OUTPUT_PATH", ""),
        "checkpoint_frequency": int(os.environ.get("CHECKPOINT_FREQUENCY", "500")),
        "num_checkpoints_to_keep": int(os.environ.get("NUM_CHECKPOINTS_TO_KEEP", "2")),
    }


class RayCheckpointCallback:
    """Callback for periodic checkpointing with Ray Train."""

    def __init__(self, checkpoint_frequency: int = 500):
        self.checkpoint_frequency = checkpoint_frequency
        self.steps = 0

    def on_step_end(self, trainer, model, tokenizer):
        self.steps += 1
        if self.steps % self.checkpoint_frequency == 0:
            self._save_checkpoint(trainer, model, tokenizer)

    def _save_checkpoint(self, trainer, model, tokenizer):
        rank = train.get_context().get_world_rank()
        if rank != 0:
            return

        logger.info(f"Saving checkpoint at step {self.steps}")

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            tokenizer.save_pretrained(tmpdir)

            import json
            state = {
                "step": self.steps,
                "trainer_state": trainer.state.save_to_json() if hasattr(trainer.state, 'save_to_json') else {},
            }
            with open(os.path.join(tmpdir, "training_state.json"), "w") as f:
                json.dump(state, f)

            checkpoint = Checkpoint.from_directory(tmpdir)
            train.report({"step": self.steps}, checkpoint=checkpoint)


def upload_to_s3(local_path: str, s3_path: str):
    """Upload a directory to S3."""
    import boto3
    from urllib.parse import urlparse

    if not s3_path.startswith("s3://"):
        logger.warning(f"Invalid S3 path: {s3_path}, skipping upload")
        return

    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            relative_path = os.path.relpath(local_file, local_path)
            s3_key = os.path.join(prefix, relative_path)

            logger.info(f"Uploading {local_file} to s3://{bucket}/{s3_key}")
            s3.upload_file(local_file, bucket, s3_key)

    logger.info(f"Upload complete: {s3_path}")


def train_func(config: Dict[str, Any]):
    """
    Per-worker training function executed on each Ray worker.

    Ray Train handles distributed setup (replaces torchrun/accelerate).
    """
    import torch
    import torch.distributed as dist
    from core import (
        TrainingConfig,
        resolve_attention,
        get_lora_config,
        load_model_and_tokenizer,
        build_fsdp_config,
        load_sft_dataset,
        create_sft_trainer,
        save_model,
    )

    # Ray Train provides distributed context
    world_size = train.get_context().get_world_size()
    rank = train.get_context().get_world_rank()
    local_rank = train.get_context().get_local_rank()

    # Debug GPU environment
    logger.info(
        f"[Worker {rank}] GPU env: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}, "
        f"device_count={torch.cuda.device_count()}, local_rank={local_rank}"
    )

    torch.cuda.set_device(local_rank)

    if rank == 0:
        logger.info(f"Starting training with {world_size} workers")
        logger.info(f"Config: {config}")

    # Initialize distributed backend (required for FSDP)
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Resolve attention implementation
    attn_impl = resolve_attention(config["attn_implementation"])
    if rank == 0:
        logger.info(f"Using attention implementation: {attn_impl}")

    # Device map: for QLoRA without FSDP, set so bitsandbytes quantizes onto the right GPU
    if config["use_qlora"] and not config["use_fsdp"]:
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = None

    if rank == 0:
        logger.info(f"Loading model: {config['model_name']}")

    model, tokenizer = load_model_and_tokenizer(
        config["model_name"],
        use_qlora=config["use_qlora"],
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        use_fsdp=config["use_fsdp"],
        device_map=device_map,
    )

    # Debug: log post-load state
    gpu_mem = torch.cuda.memory_allocated(local_rank) / 1024**3
    logger.info(f"[Worker {rank}] Model loaded. GPU mem: {gpu_mem:.2f} GB")

    lora_config = get_lora_config(
        r=config["lora_r"],
        alpha=config["lora_alpha"],
        dropout=config["lora_dropout"],
    )

    # FSDP configuration
    fsdp_str = ""
    fsdp_config_dict = None
    gradient_checkpointing = config["use_qlora"] and not config["use_fsdp"]

    if config["use_fsdp"]:
        fsdp_str, fsdp_config_dict = build_fsdp_config(model)
        gradient_checkpointing = False  # Handled by FSDP config
        if rank == 0:
            logger.info(f"FSDP wrapping classes: {fsdp_config_dict['fsdp_transformer_layer_cls_to_wrap']}")

    # Load dataset
    dataset = load_sft_dataset(config["dataset_name"], config["max_samples"])

    if rank == 0:
        logger.info(f"Dataset size: {len(dataset)}")

    # Determine DDP
    if config["use_ddp"] == "auto":
        use_ddp = config["num_workers"] > 1 and not config["use_fsdp"]
    else:
        use_ddp = config["use_ddp"] == "true"

    if rank == 0:
        logger.info(f"Workers: {config['num_workers']}, GPUs/worker: {config['gpus_per_worker']}")
        logger.info(f"DDP: {use_ddp} (setting: {config['use_ddp']}), FSDP: {config['use_fsdp']}")

    # Build TrainingConfig for create_sft_trainer
    training_config = TrainingConfig(
        model_name=config["model_name"],
        output_dir=config["output_dir"],
        batch_size=config["batch_size"],
        grad_accum_steps=config["grad_accum_steps"],
        lr=config["lr"],
        epochs=config["epochs"],
        max_seq_length=config["max_seq_length"],
        use_qlora=config["use_qlora"],
        use_fsdp=config["use_fsdp"],
        num_workers=config["num_workers"],
    )

    trainer = create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        lora_config=lora_config,
        config=training_config,
        torch_dtype=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
        fsdp_str=fsdp_str,
        fsdp_config_dict=fsdp_config_dict,
        use_ddp=use_ddp,
        local_rank=local_rank,
    )

    if rank == 0:
        logger.info("Starting training...")

    trainer.train()

    # Save model
    if rank == 0:
        output_path = config.get("output_path", "")
        local_output = config["output_dir"]
        model_short_name = config["model_name"].split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        if output_path and output_path.startswith("s3://"):
            final_s3_path = f"{output_path.rstrip('/')}/{model_short_name}/{timestamp}"
            logger.info(f"Saving model to S3: {final_s3_path}")

            with tempfile.TemporaryDirectory() as tmpdir:
                save_model(trainer, tokenizer, tmpdir, use_fsdp=config["use_fsdp"])
                upload_to_s3(tmpdir, final_s3_path)
                logger.info(f"Model uploaded to: {final_s3_path}")
        else:
            logger.info(f"Saving model locally to {local_output}")
            os.makedirs(local_output, exist_ok=True)
            save_model(trainer, tokenizer, local_output, use_fsdp=config["use_fsdp"])

    # Report final metrics to Ray
    final_loss = 0.0
    if trainer.state.log_history:
        final_loss = trainer.state.log_history[-1].get("loss", 0.0)

    train.report({"loss": final_loss, "rank": rank})

    if rank == 0:
        logger.info("Training completed!")

    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    """Main entry point for Ray Train distributed training."""
    if not ray.is_initialized():
        ray.init()

    config = get_training_config()

    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    gpus_per_worker = int(os.environ.get("GPUS_PER_WORKER", "4"))

    logger.info(f"Configuring Ray Train: {num_workers} workers, {gpus_per_worker} GPUs each")
    logger.info(f"Total GPUs: {num_workers * gpus_per_worker}")

    cpus_per_worker = config["cpus_per_worker"]
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
        resources_per_worker={
            "GPU": gpus_per_worker,
            "CPU": cpus_per_worker,
        },
    )

    storage_path = config["ray_storage_path"]
    num_checkpoints = config["num_checkpoints_to_keep"]

    if storage_path.startswith("s3://"):
        logger.info(f"Using S3 storage: {storage_path}")
        logger.info(f"Checkpoint frequency: {config['checkpoint_frequency']} steps")
        logger.info(f"Keeping {num_checkpoints} checkpoints")
    else:
        logger.info(f"Using local storage: {storage_path}")

    run_config = RunConfig(
        name="sft-training-ray",
        checkpoint_config=CheckpointConfig(
            num_to_keep=num_checkpoints,
        ),
        storage_path=storage_path,
        failure_config=FailureConfig(
            max_failures=3,
        ),
    )

    resume_from_checkpoint = None
    if storage_path.startswith("s3://"):
        try:
            from ray.train import Result
            logger.info("Checking for existing checkpoints to resume from...")
        except Exception as e:
            logger.info(f"No checkpoint found to resume from: {e}")

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        run_config=run_config,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    logger.info("Starting Ray Train job...")
    result = trainer.fit()

    logger.info(f"Training completed!")
    logger.info(f"Final metrics: {result.metrics}")
    if result.checkpoint:
        logger.info(f"Best checkpoint: {result.checkpoint}")

    output_path = config.get("output_path", "")
    if output_path and output_path.startswith("s3://"):
        model_name = config["model_name"].split("/")[-1]
        logger.info(f"Final model saved to: {output_path}/{model_name}/<timestamp>")


if __name__ == "__main__":
    main()
