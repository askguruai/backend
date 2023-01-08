from utils import CONFIG
import requests
from typing import List
import numpy as np


def get_embeddings(chunks: List[str]) -> List[np.ndarray]:
    embeddings = []
    req_data = {"input": chunks}
    response = requests.post(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/embeddings",
        json=req_data,
    ).json()
    for emb in response["data"]:
        embeddings.append(np.array(emb["embedding"]))
    return embeddings


def get_answer(context: str, query: str) -> str:
    req_data = {"info": context, "query": query}
    response = requests.post(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/completions", json=req_data
    ).json()
    return response["data"]
