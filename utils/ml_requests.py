from typing import List, Union

import numpy as np
import requests
from fastapi import status

from utils import CONFIG
from utils.errors import CoreMLError


def get_embeddings(chunks: Union[str, List[str]]) -> List[np.ndarray]:
    response = requests.post(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/embeddings",
        json={"input": chunks},
    )
    if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        raise CoreMLError(response.json()["detail"])
    embeddings = [np.array(emb["embedding"]) for emb in response.json()["data"]]
    return embeddings


def get_answer(context: str, query: str) -> str:
    response = requests.post(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/completions",
        json={"info": context, "query": query},
    )
    if response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        raise CoreMLError(response.json()["detail"])
    return response.json()["data"]
