import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Example: image-to-text
from PIL import Image
img = Image.open("/home/administrator/Romaco/langchain/lessons/1. RAG/examples/pixegami/PDF_files_langchain/rag-tutorial-v2-main/coin.jpg").convert("RGB")
vision_input = processor(images=img, return_tensors="pt").to(model.device)

prompt = "<|image_0|>"  # triggers vision input
output = model.generate(
    **vision_input,
    prompt=prompt,
    max_new_tokens=128
)
print(processor.decode(output[0]))
