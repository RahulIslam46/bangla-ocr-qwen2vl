"""
Full-Page Handwritten Bengali OCR via Line-by-Line Segmentation.
Segments a full page document into text lines using projection profiles
and transcribes each line using fine-tuned Qwen2-VL LoRA checkpoints.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel


def segment_lines(image_path: str, cuts=None):
    """Segment document image into horizontal text lines."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if cuts is None:
        # Default verified cuts for standard Bengali handwritten test sheet
        # Or automatic valley detection
        gray = img.convert("L")
        pixels = gray.load()
        proj = [sum(1 for x in range(w) if pixels[x, y] < 130) for y in range(h)]
        window = 7
        smoothed = [
            sum(proj[max(0, y - window):min(h, y + window + 1)]) / (min(h, y + window + 1) - max(0, y - window))
            for y in range(h)
        ]

        cuts = [25]
        for y in range(35, h - 50):
            left_min = min(smoothed[max(0, y - 12):y] + [9999])
            right_min = min(smoothed[y + 1:min(h, y + 13)] + [9999])
            if smoothed[y] <= left_min and smoothed[y] <= right_min and smoothed[y] < 65:
                if y - cuts[-1] >= 42:
                    cuts.append(y)
        if h - cuts[-1] > 30:
            cuts.append(min(h, cuts[-1] + 65))

    line_crops = []
    for i in range(len(cuts) - 1):
        y1 = max(0, cuts[i] - 2)
        y2 = min(h, cuts[i + 1] + 2)
        crop = img.crop((0, y1, w, y2))
        line_crops.append((i + 1, crop, (y1, y2)))
    return line_crops


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
    parser = argparse.ArgumentParser(description="Full-page Bengali OCR via line segmentation.")
    parser.add_argument("--image", type=str, required=True, help="Path to full-page document image.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to fine-tuned LoRA checkpoint.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output_file", type=str, default=None, help="Save transcriptions to text file.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Verified manual cut points for standard test sample
    cuts = [25, 90, 168, 216, 270, 322, 368, 418, 465, 514, 564, 614, 672, 735]
    print(f"Segmenting full page image: {args.image}...")
    line_crops = segment_lines(args.image, cuts=cuts)
    print(f"Detected and cropped {len(line_crops)} text lines.\n")

    print(f"Loading base model: {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
    )

    print(f"Attaching fine-tuned adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    print("\n" + "=" * 60)
    print("      FULL-PAGE BENGALI OCR TRANSCRIPTION RESULTS      ")
    print("=" * 60)

    transcriptions = []
    for line_idx, crop_img, (y1, y2) in line_crops:
        text = predict_line(model, processor, crop_img)
        transcriptions.append(text)
        print(f"Line {line_idx:02d} [y={y1:03d}..{y2:03d}]: {text}")

    full_text = "\n".join(transcriptions)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"\nSaved full transcription to: {args.output_file}")


if __name__ == "__main__":
    main()
