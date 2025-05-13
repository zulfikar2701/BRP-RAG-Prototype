from vllm import LLM, SamplingParams
import torch
from .config import *

_llm = None  # lazy init: keeps GPU free until needed

def answer(question, contexts):
    global _llm
    if _llm is None:
        torch.cuda.empty_cache()
        _llm = LLM(model=LLM_MODEL, dtype=torch.bfloat16,
                   gpu_memory_utilization=0.85, max_model_len=1024)
    ctx_txt = " ".join(reversed([c["entity"]["chunk"] for c in contexts]))
    src_txt = ", ".join({c['entity']['source'] for c in contexts})
    prompt  = (
        f"Context: {ctx_txt}\nSources: {src_txt}\n\n"
        f"User: {question}\nAssistant:"
    )
    out = _llm.generate([prompt], SamplingParams(
    temperature=0.2,
    top_p=0.95,
    max_tokens=512,     # default was 256 – bump to 512
))
    return out[0].outputs[0].text.strip()
