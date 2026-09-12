"""
Dataset loader and preprocessor for BN-HTRd dataset with Qwen2-VL.
Supports both line-level OCR and page-level document OCR modes.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


def parse_bn_htrd(
    data_dir: str | Path,
    mode: str = "line",
    max_samples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Parse BN-HTRd dataset directory into structured samples.
    
    Args:
        data_dir: Root directory containing document folders (e.g. '1', '8', '50', '100').
        mode: 'line' for cropped line images, 'page' for full document page images.
        max_samples: Optional maximum number of samples to return (useful for quick testing).
        
    Returns:
        List of dicts: [{"id": str, "image_path": str, "text": str}]
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    samples = []
    
    # Iterate through each document folder
    for doc_dir in sorted(data_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
            
        xlsx_files = list(doc_dir.glob("*.xlsx"))
        if not xlsx_files:
            continue
            
        xlsx_path = xlsx_files[0]
        try:
            df = pd.read_excel(xlsx_path)
        except Exception as e:
            print(f"Warning: Could not read {xlsx_path}: {e}")
            continue

        if "Id" not in df.columns or "Word" not in df.columns:
            continue

        if mode == "line":
            # Group words by line: doc_page_line
            line_groups: Dict[str, List[Tuple[int, str]]] = {}
            for _, row in df.iterrows():
                raw_id = str(row["Id"]).strip()
                word = str(row["Word"]).strip() if pd.notna(row["Word"]) else ""
                parts = raw_id.split("_")
                if len(parts) == 4:
                    doc_id, page_id, line_id, word_id = parts
                    try:
                        w_idx = int(word_id)
                    except ValueError:
                        w_idx = 0
                    line_key = f"{doc_id}_{page_id}_{line_id}"
                    line_groups.setdefault(line_key, []).append((w_idx, word))

            for line_key, words_list in line_groups.items():
                parts = line_key.split("_")
                doc_id, page_id, line_id = parts[0], parts[1], parts[2]
                img_path = doc_dir / "Lines" / f"{doc_id}_{page_id}" / f"{line_key}.jpg"
                
                if not img_path.exists():
                    continue

                sorted_words = [w for _, w in sorted(words_list, key=lambda x: x[0]) if w]
                text = " ".join(sorted_words).strip()
                if not text:
                    continue

                samples.append({
                    "id": line_key,
                    "image_path": str(img_path),
                    "text": text,
                })
                
                if max_samples and len(samples) >= max_samples:
                    return samples

        elif mode == "page":
            # Group words by page and line: doc_page -> line_id -> words
            page_groups: Dict[str, Dict[int, List[Tuple[int, str]]]] = {}
            for _, row in df.iterrows():
                raw_id = str(row["Id"]).strip()
                word = str(row["Word"]).strip() if pd.notna(row["Word"]) else ""
                parts = raw_id.split("_")
                if len(parts) == 4:
                    doc_id, page_id, line_id, word_id = parts
                    try:
                        l_idx = int(line_id)
                        w_idx = int(word_id)
                    except ValueError:
                        continue
                    page_key = f"{doc_id}_{page_id}"
                    page_groups.setdefault(page_key, {}).setdefault(l_idx, []).append((w_idx, word))

            for page_key, lines_dict in page_groups.items():
                img_path = doc_dir / f"{page_key}.jpg"
                if not img_path.exists():
                    continue

                # Build page text preserving line breaks
                page_lines = []
                for l_idx in sorted(lines_dict.keys()):
                    sorted_words = [w for _, w in sorted(lines_dict[l_idx], key=lambda x: x[0]) if w]
                    line_text = " ".join(sorted_words).strip()
                    if line_text:
                        page_lines.append(line_text)

                text = "\n".join(page_lines).strip()
                if not text:
                    continue

                samples.append({
                    "id": page_key,
                    "image_path": str(img_path),
                    "text": text,
                })

                if max_samples and len(samples) >= max_samples:
                    return samples
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'line' or 'page'.")

    return samples


class BNHTRdDataset(Dataset):
    """PyTorch Dataset for BN-HTRd Bengali Handwritten Text Recognition."""

    def __init__(
        self,
        data_dir: str | Path,
        mode: str = "line",
        max_samples: Optional[int] = None,
        samples: Optional[List[Dict[str, str]]] = None,
    ):
        super().__init__()
        self.mode = mode
        if samples is not None:
            self.samples = samples
        else:
            self.samples = parse_bn_htrd(data_dir=data_dir, mode=mode, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "id": item["id"],
            "image": image,
            "text": item["text"],
            "image_path": item["image_path"],
        }


def create_collate_fn(
    processor,
    max_image_size: Tuple[int, int] = (640, 640),
    prompt_text: str = "Extract the handwritten Bengali text from this image accurately.",
):
    """
    Collate function to prepare vision and text inputs for Qwen2-VL.
    Labels are created such that prompt tokens are masked with -100 for loss computation.
    """
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = []
        texts = []

        for item in batch:
            img = item["image"].copy()
            # Constrain image dimensions to keep visual token count bounded
            img.thumbnail(max_image_size, Image.Resampling.LANCZOS)
            images.append(img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                },
            ]
            full_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            target_text = item["text"]
            # Complete conversation text ending with eos token
            full_text = f"{full_prompt}{target_text}<|im_end|>"
            texts.append((full_prompt, full_text))

        # Process batch through Qwen2-VL processor
        prompt_texts = [p for p, _ in texts]
        full_texts = [f for _, f in texts]

        inputs = processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()

        # Mask prompt tokens so loss is only computed on the target output
        for i, (prompt, _) in enumerate(texts):
            prompt_token_ids = processor.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_token_ids)
            labels[i, :prompt_len] = -100

        # Mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs

    return collate_fn
