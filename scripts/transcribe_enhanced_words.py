"""
Evaluate fine-tuned Qwen2-VL OCR on the user's enhanced & segmented word dataset.
Compares model predictions directly against ground-truth metadata.
"""

import os
import re
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


def predict_single_word(model, processor, image: Image.Image, prompt_text="Extract the handwritten text from this image accurately.") -> str:
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
            max_new_tokens=15,  # strictly bounded to word length to prevent runaway hallucination
            do_sample=False,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate on enhanced segmented words.")
    parser.add_argument("--words_dir", type=str, default="/content/bangla-ocr-qwen2vl/data/research_words_enhanced")
    parser.add_argument("--adapter_path", type=str, default="/content/drive/MyDrive/bangla_ocr_checkpoints/checkpoint-2600")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load metadata
    meta_path = os.path.join(args.words_dir, "metadata.json")
    ground_truths = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            for item in meta:
                ground_truths[item["id"]] = item["text"]

    # Collect word images in numerical order
    files = [f for f in os.listdir(args.words_dir) if f.startswith("word") and f.endswith(".png")]
    files.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    print(f"Found {len(files)} words to evaluate.")

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

    print("\n" + "=" * 70)
    print("      ENHANCED WORD-BY-WORD OCR EVALUATION RESULTS      ")
    print("=" * 70)
    print(f"{'#':<4} | {'File':<12} | {'Ground Truth':<16} | {'Model Prediction'}")
    print("-" * 70)

    full_prediction = []
    for fname in files:
        w_id = int(re.search(r"\d+", fname).group())
        img_path = os.path.join(args.words_dir, fname)
        img = Image.open(img_path).convert("RGB")
        pred = predict_single_word(model, processor, img)
        gt = ground_truths.get(w_id, "")
        full_prediction.append(pred)
        print(f"{w_id:<4} | {fname:<12} | {gt:<16} | {pred}")

    print("=" * 70)
    print("\nFull Assembled Text:")
    print(" ".join(full_prediction))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
