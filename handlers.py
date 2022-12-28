import hashlib
import json
import os.path as osp
from pathlib import Path

import numpy as np
import requests

from parsers import parse_text
from utils import CONFIG
from utils.data import TextRequest


def text_request_handler(request: TextRequest) -> str:
    input_hash = hashlib.sha256(request.text_input.encode()).hexdigest()
    cachefile_path = Path(osp.join(CONFIG["app"]["storage_path"], f"{input_hash}.json"))
    if cachefile_path.exists():
        with open(cachefile_path, "rt") as f:
            input_data = json.load(f)
        input_embeddings = []
        for chunk in input_data:
            input_embeddings.append(chunk["embedding"])
        input_embeddings = np.array(input_embeddings)
        req_data = {
            "input": request.query
        }
        response = requests.post(f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/embeddings",
                                 json=req_data).json()
        query_embedding = np.array(response["data"][0]["embedding"])
    else:
        input_chunks = parse_text(request.text_input)
        req_data = {
            "input": input_chunks + [request.query]
        }
        response = requests.post(f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/embeddings",
                                 json=req_data).json()
        input_data = []
        input_embeddings = []
        for i, emb in enumerate(response["data"][:-1]):
            input_embeddings.append(emb["embedding"])
            input_data.append({
                "text": input_chunks[i],
                "embedding": emb["embedding"]
            })
        assert len(input_data) == len(input_chunks)  # todo: make global error handling
        with open(cachefile_path, "wt") as f:
            json.dump(input_data, f)
        input_embeddings = np.array(input_embeddings)
        query_embedding = np.array(response["data"][-1]["embedding"])

    cosines = [np.dot(emb, query_embedding) for emb in input_embeddings]
    indices = np.argsort(cosines)[-3:]  # picking top 3 relevant results. TODO: make a config parameter
    context = " ".join([input_data[i]["text"] for i in indices])
    req_data = {
        "info": context,
        "query": request.query
    }
    response = requests.post(f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}/completions",
                             json=req_data).json()
    return response["data"]
