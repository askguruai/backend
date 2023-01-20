from typing import List, Union

import numpy as np
import requests

from utils import CONFIG
from utils.errors import CoreMLError


def get_embeddings(chunks: Union[str, List[str]]) -> List[np.ndarray]:
    embeddings = []
    req_data = {"input": chunks}
    response = requests.post(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/embeddings",
        json=req_data,
    ).json()
    if response["status"] == "error":
        raise CoreMLError(f"CoreML error: {response['message']}")
    for emb in response["data"]:
        embeddings.append(np.array(emb["embedding"]))
    return embeddings


def get_answer(context: str, query: str) -> str:
    req_data = {"info": context, "query": query}
    response = requests.post(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/completions", json=req_data
    ).json()
    if response["status"] == "error":
        raise CoreMLError(f"CoreML error: {response['message']}")
    return response["data"]
