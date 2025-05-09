from langchain.llms import VLLM
import time
import csv


llm = VLLM(model="tiiuae/falcon-7b-instruct",
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=50,
           temperature=0.6
)


start_time = time.time()
output = llm("Who is president of Indonesia?")
end_time = time.time()
latency = end_time - start_time
print(f"Latency: {latency} seconds")
print("Generated text:", output)