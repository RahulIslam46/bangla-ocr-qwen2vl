"""
Evaluate fine-tuned Qwen2-VL OCR on full handwritten lines.
Evaluates both line-by-line strips and 2-line chunks to preserve grammatical context.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


GROUND_TRUTH = {
    "line1_plobota.png": "প্লবতা, F = Ahρg",
    "line2_ekhane.png": "এখানে h বস্তুর উচ্চতা নাকি",
    "line3_panir.png": "পানির মুক্ততল থেকে বস্তুর",
    "line4_toldesh.png": "তলদেশ পর্যন্ত উচ্চতা?",
    "part1_two_lines.png": "প্লবতা, F = Ahρg\nএখানে h বস্তুর উচ্চতা নাকি",
    "part2_two_lines.png": "পানির মুক্ততল থেকে বস্তুর\nতলদেশ পর্যন্ত উচ্চতা?",
}


def predict_line(model, processor, image: Image.Image, prompt_text="Extract the handwritten text from this image accurately.") -> str:
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
            max_new_tokens=64,
            do_sample=False,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate on full handwritten lines.")
    parser.add_argument("--lines_dir", type=str, default="/content/bangla-ocr-qwen2vl/data/enhanced_lines")
    parser.add_argument("--adapter_path", type=str, default="/content/drive/MyDrive/bangla_ocr_checkpoints/checkpoint-2600")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading processor & base model: {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # 1. Evaluate Individual Lines (4 lines)
    individual_lines = [
        "line1_plobota.png",
        "line2_ekhane.png",
        "line3_panir.png",
        "line4_toldesh.png",
    ]

    print("\n" + "=" * 80)
    print("       FULL-LINE HANDWRITTEN OCR EVALUATION (LINE BY LINE)       ")
    print("=" * 80)
    print(f"{'#':<3} | {'Line File':<20} | {'Ground Truth':<32} | {'Model Prediction'}")
    print("-" * 80)

    for i, fname in enumerate(individual_lines, 1):
        fpath = os.path.join(args.lines_dir, fname)
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        pred = predict_line(model, processor, img)
        gt = GROUND_TRUTH.get(fname, "")
        print(f"{i:<3} | {fname:<20} | {gt:<32} | {pred}")

    print("=" * 80 + "\n")

    # 2. Evaluate 2 Full Multi-Line Chunks
    two_line_files = [
        "part1_two_lines.png",
        "part2_two_lines.png",
    ]

    print("=" * 80)
    print("       2-LINE COMBINED CHUNK OCR EVALUATION (MULTI-LINE)         ")
    print("=" * 80)
    for fname in two_line_files:
        fpath = os.path.join(args.lines_dir, fname)
        if not os.path.exists(fpath):
            continue
        img = Image.open(fpath).convert("RGB")
        pred = predict_line(model, processor, img)
        gt = GROUND_TRUTH.get(fname, "").replace("\n", " ")
        print(f"\n[File: {fname}]")
        print(f"  Ground Truth: {gt}")
        print(f"  Prediction:   {pred}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
