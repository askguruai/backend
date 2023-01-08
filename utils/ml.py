from utils import CONFIG
import requests
from typing import List, Dict, Union
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


def get_context_indices(chunk_embeddings: Union[List, np.ndarray],
                        query_embedding: Union[List, np.ndarray]) -> np.ndarray:
    query_embedding = np.array(query_embedding)
    cosines = [np.dot(emb, query_embedding) for emb in chunk_embeddings]
    return np.argsort(cosines)[-int(CONFIG["text_handler"]["top_k_chunks"]):][::-1]
