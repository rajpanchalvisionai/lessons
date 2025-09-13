from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Initialize vLLM and processor once (global, for efficiency)
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    gpu_memory_utilization=0.85,
    max_model_len=100_000,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    cpu_offload_gb=4,
    enable_chunked_prefill=True
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

def inference(image_path: str, prompt_text: str = "What do you see?") -> str:
    """
    Run inference on the given image and prompt text.
    Args:
        image_path (str): Path to the image file.
        prompt_text (str): The prompt/question to ask about the image.
    Returns:
        str: The generated response from the model.
    """
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    formatted_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    jobs = [
        {
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": img}
        }
    ]
    outputs = llm.generate(jobs, SamplingParams(temperature=0.2, max_tokens=128))
    return outputs[0].outputs[0].text

# Optional: test block for standalone execution
if __name__ == "__main__":
    result = inference(
        "lessons/1. RAG/examples/pixegami/PDF_files_langchain/rag-tutorial-v2-main/coin.jpg",
        "What do you see?"
    )
    print(result)