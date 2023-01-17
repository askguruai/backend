from typing import List, Union

import numpy as np
import requests

from utils import CONFIG


def get_embeddings(chunks: Union[str, List[str]]) -> List[np.ndarray]:
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


def get_context_from_chunks_embeddings(
    chunks: List[str], embeddings: List[List[float]], query: str, top_k_chunks: int = 3
) -> tuple[str, np.ndarray]:
    query_embedding = get_embeddings(query)[0]
    distances = [np.dot(embedding, query_embedding) for embedding in embeddings]
    indices = np.argsort(distances)[-top_k_chunks:][::-1]
    context = "\n\n".join([chunks[i] for i in indices])
    return context, indices
