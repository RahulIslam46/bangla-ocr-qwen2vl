"""
Transcribe a directory of pre-cropped text lines with fine-tuned Qwen2-VL.
Loads the model once and evaluates all crops sequentially.
"""

import os
import argparse
import glob
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


def predict_line(model, processor, image: Image.Image, prompt_text="Extract the handwritten Bengali text from this image accurately.") -> str:
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

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe line crops.")
    parser.add_argument("--crops_dir", type=str, required=True, help="Directory containing line crops.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA checkpoint.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    crop_files = sorted([f for f in glob.glob(os.path.join(args.crops_dir, "*.png")) if not os.path.basename(f).startswith('.')])
    if not crop_files:
        raise FileNotFoundError(f"No crop files found in {args.crops_dir}")

    print(f"Found {len(crop_files)} line crops in {args.crops_dir}.")

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

    print("\n" + "=" * 60)
    print("      LINE-BY-LINE OCR TRANSCRIPTION RESULTS      ")
    print("=" * 60)

    for i, path in enumerate(crop_files, 1):
        img = Image.open(path).convert("RGB")
        pred = predict_line(model, processor, img)
        fname = os.path.basename(path)
        print(f"Line {i:02d} ({fname}): {pred}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
