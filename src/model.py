"""
Model loader and QLoRA configuration for Qwen2-VL.
Configured for efficient fine-tuning on consumer/free-tier GPUs (e.g., Google Colab T4).
"""

from typing import Tuple, List, Optional
import torch


def get_target_modules() -> List[str]:
    """Target modules for LoRA in Qwen2-VL architecture."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def print_trainable_parameters(model) -> None:
    """Print the number and percentage of trainable parameters."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    pct = 100 * trainable_params / all_params if all_params > 0 else 0
    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable %: {pct:.2f}%"
    )


def load_qwen2_vl_model(
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    use_qlora: bool = True,
    device: str = "cuda",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = True,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 640 * 640,
):
    """
    Load Qwen2-VL model and processor with optional 4-bit NF4 QLoRA quantization.

    Args:
        model_id: Hugging Face model identifier.
        use_qlora: Whether to apply 4-bit NF4 quantization + LoRA adapters.
        device: 'cuda' or 'cpu'.
        lora_r: Rank of LoRA adapters.
        lora_alpha: Scaling factor for LoRA.
        lora_dropout: Dropout probability for LoRA layers.
        gradient_checkpointing: Enable gradient checkpointing for VRAM savings.
        min_pixels: Minimum resolution limit for processor.
        max_pixels: Maximum resolution limit for processor.

    Returns:
        (model, processor)
    """
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    if use_qlora:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("Configuring 4-bit NF4 quantization...")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        print(f"Loading model {model_id} in 4-bit...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=compute_dtype,
        )

        if gradient_checkpointing:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )
            model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=get_target_modules(),
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        print("PEFT QLoRA adapter attached:")
        print_trainable_parameters(model)

    else:
        print(f"Loading base model {model_id} without quantization...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        )

    return model, processor
