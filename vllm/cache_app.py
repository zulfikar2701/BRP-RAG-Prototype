import langchain
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import FastAPI
from langchain.llms import VLLM
import time
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
import hashlib
import hashlib
import uvicorn

app = FastAPI()


def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


langchain.llm_cache = GPTCache(init_gptcache)
llm = VLLM(model="tiiuae/falcon-7b-instruct",
           trust_remote_code=True,  # mandatory for hf models
           max_new_tokens=50,
           temperature=0.6
)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/v1/generateText")
async def generateText(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    print(prompt)
    output = llm(prompt)
    print("Generated text:", output)
    ret = {"text": output}
    return JSONResponse(ret)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)