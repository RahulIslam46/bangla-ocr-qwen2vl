"""
Evaluation and inference script for Bengali Handwritten OCR with Qwen2-VL.
Calculates Character Error Rate (CER) and Word Error Rate (WER) using jiwer.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import jiwer

from src.dataset import parse_bn_htrd


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Bengali OCR with Qwen2-VL.")
    parser.add_argument("--image", type=str, default=None, help="Path to a single test image.")
    parser.add_argument("--data_dir", type=str, default=None, help="Evaluate across dataset directory.")
    parser.add_argument("--mode", type=str, default="line", choices=["line", "page"])
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to fine-tuned LoRA adapter directory.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate if evaluating data_dir.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def predict_ocr(model, processor, image: Image.Image, prompt_text: str = "Extract the handwritten Bengali text from this image accurately.") -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Trim prompt tokens from generation
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load processor and base model
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if args.adapter_path and Path(args.adapter_path).exists():
        print(f"Loading LoRA adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        prediction = predict_ocr(model, processor, image)
        print(f"\nImage: {img_path.name}")
        print(f"Prediction: {prediction}")
        return

    if args.data_dir:
        samples = parse_bn_htrd(args.data_dir, mode=args.mode, max_samples=args.num_samples)
        predictions = []
        references = []

        print(f"\nEvaluating on {len(samples)} samples...")
        for i, item in enumerate(samples):
            image = Image.open(item["image_path"]).convert("RGB")
            pred = predict_ocr(model, processor, image)
            gt = item["text"]
            predictions.append(pred)
            references.append(gt)
            print(f"[{i+1}/{len(samples)}] ID: {item['id']}")
            print(f"  Ground Truth: {gt}")
            print(f"  Prediction:   {pred}")

        cer = jiwer.cer(references, predictions)
        wer = jiwer.wer(references, predictions)
        print("=" * 40)
        print(f"Character Error Rate (CER): {cer * 100:.2f}%")
        print(f"Word Error Rate (WER):      {wer * 100:.2f}%")
        print("=" * 40)


if __name__ == "__main__":
    main()
