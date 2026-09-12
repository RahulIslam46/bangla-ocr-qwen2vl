"""
Training script for Bengali Handwritten OCR using Qwen2-VL and QLoRA.
Designed to run efficiently on Google Colab (Tesla T4) or any CUDA GPU.
"""

import os
import argparse
from pathlib import Path
import random
from src.dataset import BNHTRdDataset, parse_bn_htrd, create_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2-VL on BN-HTRd dataset with QLoRA.")
    parser.add_argument("--data_dir", type=str, default="./data/sample", help="Path to BN-HTRd data directory.")
    parser.add_argument("--mode", type=str, default="line", choices=["line", "page"], help="Training mode: 'line' crops or full 'page'.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Where to save model checkpoints.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Base model identifier.")
    parser.add_argument("--epochs", type=int, default=3, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective batch = batch_size * grad_accum).")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio for cosine scheduler.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of samples for validation.")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total samples (useful for quick debug).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X steps.")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log metrics every X steps.")
    parser.add_argument("--max_image_dim", type=int, default=640, help="Max width/height for input images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from transformers import TrainingArguments, Trainer
        from src.model import load_qwen2_vl_model
    except ImportError as e:
        print(f"\n[Error] Missing training dependency: {e}")
        print("Please ensure dependencies from requirements.txt are installed:")
        print("  pip install -r requirements.txt\n")
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print(" Bengali OCR Training Pipeline with Qwen2-VL (QLoRA) ")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model ID: {args.model_id}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")
    print("=" * 60)

    # 1. Parse dataset
    all_samples = parse_bn_htrd(
        data_dir=args.data_dir,
        mode=args.mode,
        max_samples=args.max_samples,
    )
    if not all_samples:
        raise RuntimeError(f"No valid samples found in {args.data_dir} for mode '{args.mode}'!")

    print(f"Total parsed samples: {len(all_samples)}")
    random.shuffle(all_samples)

    val_count = max(1, int(len(all_samples) * args.val_split))
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]
    print(f"Train samples: {len(train_samples)} | Validation samples: {len(val_samples)}")

    # 2. Load Model & Processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA is not available. Running in CPU mode (slow, test-only).")

    model, processor = load_qwen2_vl_model(
        model_id=args.model_id,
        use_qlora=(device == "cuda"),
        device=device,
        max_pixels=args.max_image_dim * args.max_image_dim,
    )

    # 3. Create Datasets & Collate Function
    train_dataset = BNHTRdDataset(data_dir=args.data_dir, samples=train_samples)
    val_dataset = BNHTRdDataset(data_dir=args.data_dir, samples=val_samples)
    collate_fn = create_collate_fn(
        processor=processor,
        max_image_size=(args.max_image_dim, args.max_image_dim),
    )

    # 4. Configure Training Arguments with dynamic version compatibility
    use_fp16 = device == "cuda" and not torch.cuda.is_bf16_supported()
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()

    import inspect
    sig = inspect.signature(TrainingArguments.__init__).parameters

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "num_train_epochs": args.epochs,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": "paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "report_to": "none",
    }

    if "warmup_steps" in sig:
        training_kwargs["warmup_steps"] = 20
    elif "warmup_ratio" in sig:
        training_kwargs["warmup_ratio"] = args.warmup_ratio

    if len(val_samples) > 0:
        if "eval_strategy" in sig:
            training_kwargs["eval_strategy"] = "steps"
            training_kwargs["eval_steps"] = args.save_steps
        elif "evaluation_strategy" in sig:
            training_kwargs["evaluation_strategy"] = "steps"
            training_kwargs["eval_steps"] = args.save_steps

    training_args = TrainingArguments(**training_kwargs)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_samples) > 0 else None,
        data_collator=collate_fn,
    )

    # 6. Execute Training
    print("\nStarting training loop...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 7. Save Final LoRA Weights & Processor
    final_save_dir = Path(args.output_dir) / "final_model"
    final_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTraining completed! Saving LoRA adapter to {final_save_dir}...")
    model.save_pretrained(str(final_save_dir))
    processor.save_pretrained(str(final_save_dir))
    print("All done!")


if __name__ == "__main__":
    main()
