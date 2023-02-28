from typing import List, Union

import numpy as np
import requests
from fastapi import status

from utils import CONFIG
from utils.errors import CoreMLError


def get_embeddings(chunks: List[str], api_version: str) -> List[np.ndarray]:
    if type(chunks) is not list:
        chunks = [chunks]
    response = requests.post(
        f"{CONFIG['coreml']['route']}/{api_version}/embeddings/",
        json={"input": chunks},
        timeout=20.0,
    )
    if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        raise CoreMLError(response.json()["detail"])
    embeddings = [np.array(emb) for emb in response.json()["data"]]
    return embeddings


def get_answer(context: str, query: str, api_version: str) -> str:
    response = requests.post(
        f"{CONFIG['coreml']['route']}/{api_version}/completions/",
        json={"info": context, "query": query},
        timeout=20.0,
    )
    if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        raise CoreMLError(response.json()["detail"])
    return response.json()["data"]
