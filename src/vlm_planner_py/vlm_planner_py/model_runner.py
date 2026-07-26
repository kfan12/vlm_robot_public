import torch

from PIL import Image
import time


def cv2_to_pil(bgr_img):
    """Convert OpenCV BGR numpy array to PIL Image."""
    import cv2
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# --- Qwen2.5-VL-3B (4-bit) — symbolic sign reading -------------------------
# The sign-class task needs Qwen, not SmolVLM (memory: "Qwen reads symbolic
# signs", "SmolVLM inadequate for geometry"; validated offline in
# scripts/probe_signs_world.py). Lazy-loaded on the first VLM tick.
_qwen_model = None
_qwen_processor = None
_QWEN_DEFAULT = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_qwen_model(model_name: str = _QWEN_DEFAULT, device: str = 'cuda'):
    """Lazy-load Qwen2.5-VL-3B, 4-bit nf4 on GPU (~2.5 GB on the 6 GB 3060) when
    bitsandbytes is available, else full precision on CPU. Cached module-globally
    so repeated calls are free. Logs VRAM after the first load."""
    global _qwen_model, _qwen_processor
    if _qwen_model is not None:
        return _qwen_model, _qwen_processor

    from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig)

    load_kwargs = {}
    if device == 'cuda' and torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            load_kwargs.update(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                ),
                device_map="cuda",
            )
            print("Loading 4-bit quantized Qwen on GPU.")
        except ImportError:
            load_kwargs.update(dtype=torch.float32, device_map="cpu")
            print("bitsandbytes missing -> Qwen full precision on CPU (slow).")
    else:
        load_kwargs.update(dtype=torch.float32, device_map="cpu")
        print("CUDA unavailable -> Qwen on CPU (slow).")

    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    _qwen_processor = AutoProcessor.from_pretrained(model_name)
    _qwen_model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"Qwen loaded: {model_name}. VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB "
              f"(peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")
    return _qwen_model, _qwen_processor


def run_qwen_inference(model, processor, image_pil: Image.Image, prompt_text: str,
                       max_new_tokens: int = 64):
    """Single Qwen2.5-VL inference on an in-memory PIL image. Returns
    (response_str, elapsed_sec). Greedy decode for determinism."""
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_pil},
            {"type": "text", "text": prompt_text},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    elapsed = time.time() - t0
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()
    return response, elapsed

