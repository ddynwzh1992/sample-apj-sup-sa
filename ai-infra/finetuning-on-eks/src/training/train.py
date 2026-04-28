"""
Standalone training script for SFT fine-tuning.

Supports single-GPU and torchrun distributed training.
For Ray-based distributed training, use train_ray.py instead.
"""
import os
import argparse
import logging

import torch

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0 = all)")
    args = parser.parse_args()

    # Check if running via torchrun
    is_distributed = "WORLD_SIZE" in os.environ

    if is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist
        dist.init_process_group("nccl")
        if int(os.environ["RANK"]) == 0:
            print(f"Distributed Mode: Enabled (Rank 0/{os.environ['WORLD_SIZE']})")
    else:
        print("Distributed Mode: Disabled (Single GPU)")

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn_impl = resolve_attention(args.attn_implementation)
    logger.info(f"Using attention implementation: {attn_impl}")

    config = TrainingConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        use_qlora=args.use_qlora,
        use_fsdp=args.use_fsdp,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
    )

    model, tokenizer = load_model_and_tokenizer(
        config.model_name,
        use_qlora=config.use_qlora,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
        use_fsdp=config.use_fsdp,
    )

    lora_config = get_lora_config(r=args.lora_r)

    fsdp_str = ""
    fsdp_config_dict = None
    gradient_checkpointing = config.use_qlora and not config.use_fsdp

    if config.use_fsdp:
        fsdp_str, fsdp_config_dict = build_fsdp_config(model)
        gradient_checkpointing = False  # Handled by FSDP config

    dataset = load_sft_dataset(config.dataset_name, config.max_samples)

    trainer = create_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        lora_config=lora_config,
        config=config,
        torch_dtype=torch_dtype,
        gradient_checkpointing=gradient_checkpointing,
        fsdp_str=fsdp_str,
        fsdp_config_dict=fsdp_config_dict,
    )

    trainer.train()

    save_model(trainer, tokenizer, config.output_dir, use_fsdp=config.use_fsdp)

    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
