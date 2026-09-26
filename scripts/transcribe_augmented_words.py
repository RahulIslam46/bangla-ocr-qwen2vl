"""
Evaluate fine-tuned Qwen2-VL OCR on augmented word variations.
Compares different augmentation styles (gelpen thick, ballpen thin, slanted, tremor, faded)
against ground truth to evaluate style robustness.
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


STYLE_MAP = {
    "0": "0_clean_enhanced.png",
    "clean": "0_clean_enhanced.png",
    "1": "1_gelpen_thick.png",
    "gelpen": "1_gelpen_thick.png",
    "thick": "1_gelpen_thick.png",
    "2": "2_ballpen_thin.png",
    "ballpen": "2_ballpen_thin.png",
    "thin": "2_ballpen_thin.png",
    "3": "3_slanted.png",
    "slanted": "3_slanted.png",
    "4": "4_elastic_tremor.png",
    "tremor": "4_elastic_tremor.png",
    "5": "5_degraded_faded.png",
    "faded": "5_degraded_faded.png",
}


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
            max_new_tokens=15,
            do_sample=False,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate on augmented words across styles.")
    parser.add_argument("--augmented_dir", type=str, default="/content/bangla-ocr-qwen2vl/data/research_words_augmented")
    parser.add_argument("--adapter_path", type=str, default="/content/drive/MyDrive/bangla_ocr_checkpoints/checkpoint-2600")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--styles", type=str, default="thick,thin,slanted", help="Comma-separated styles or 'all'")
    parser.add_argument("--words", type=str, default="all", help="Comma-separated word IDs or 'all'")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load metadata
    meta_path = os.path.join(args.augmented_dir, "metadata.json")
    ground_truths = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            for item in meta:
                ground_truths[item["id"]] = item["text"]

    # Resolve styles
    if args.styles.lower() == "all":
        selected_styles = [
            ("Clean", "0_clean_enhanced.png"),
            ("GelPen Thick", "1_gelpen_thick.png"),
            ("BallPen Thin", "2_ballpen_thin.png"),
            ("Slanted", "3_slanted.png"),
            ("Tremor", "4_elastic_tremor.png"),
            ("Faded", "5_degraded_faded.png"),
        ]
    else:
        selected_styles = []
        for s in args.styles.split(","):
            s = s.strip().lower()
            if s in STYLE_MAP:
                fname = STYLE_MAP[s]
                label = fname.replace(".png", "").split("_", 1)[1].replace("_", " ").title()
                selected_styles.append((label, fname))

    print(f"Styles to evaluate: {[s[0] for s in selected_styles]}")

    # Resolve words
    word_dirs = [d for d in os.listdir(args.augmented_dir) if os.path.isdir(os.path.join(args.augmented_dir, d)) and d.startswith("word")]
    word_dirs.sort(key=lambda x: int(re.search(r"\d+", x).group()))

    if args.words.lower() != "all":
        req_ids = {int(x.strip()) for x in args.words.split(",")}
        word_dirs = [d for d in word_dirs if int(re.search(r"\d+", d).group()) in req_ids]

    print(f"Evaluating {len(word_dirs)} words across {len(selected_styles)} styles ({len(word_dirs) * len(selected_styles)} images total)...")

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

    print("\n" + "=" * 95)
    print("      AUGMENTED WORD-BY-WORD OCR MULTI-STYLE EVALUATION      ")
    print("=" * 95)

    header = f"{'#':<3} | {'Ground Truth':<12} | " + " | ".join([f"{s[0]:<16}" for s in selected_styles])
    print(header)
    print("-" * len(header))

    results = []
    for wdir in word_dirs:
        w_id = int(re.search(r"\d+", wdir).group())
        gt = ground_truths.get(w_id, "")
        row_preds = []
        for s_label, s_fname in selected_styles:
            img_path = os.path.join(args.augmented_dir, wdir, s_fname)
            if not os.path.exists(img_path):
                row_preds.append("MISSING")
                continue
            img = Image.open(img_path).convert("RGB")
            pred = predict_single_word(model, processor, img)
            row_preds.append(pred)

        row_str = f"{w_id:<3} | {gt:<12} | " + " | ".join([f"{p:<16}" for p in row_preds])
        print(row_str)
        results.append({"id": w_id, "gt": gt, "preds": row_preds})

    print("=" * len(header) + "\n")


if __name__ == "__main__":
    main()
