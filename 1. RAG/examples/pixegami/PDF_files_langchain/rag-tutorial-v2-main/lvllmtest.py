from langchain_community.llms import VLLM

llm = VLLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    vllm_kwargs={
        "gpu_memory_utilization": 0.85,
        "max_model_len": 100_000,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 1024,
        "cpu_offload_gb": 4,
        "enable_chunked_prefill": True,
    },
)

print(llm.invoke("What is the capital of France ?"))