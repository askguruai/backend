import hashlib
import json
import logging
import os.path as osp
from pathlib import Path
from pprint import pformat

import numpy as np
import requests
from utils.data import TextQueryRequest

from parsers import parse_text
from utils import CONFIG, STORAGE
from utils.ml import get_embeddings, get_answer
from typing import Tuple, Any
from db import DB


def text_query_handler(data: TextQueryRequest) -> Tuple[str, str, Any]:
    info = data.text if data.text else ""
    if info == "":
        context = ""
    else:
        processed_data = None
        if data.document_id is not None and data.document_id in STORAGE:
            # checking if doc_id presented and exists in storage
            processed_data = STORAGE[data.document_id]
            doc_id = data.document_id
        else:
            # otherwise, calculating hash and checking storage again
            info_hash = STORAGE.get_hash(info)
            if info_hash in STORAGE:
                processed_data = STORAGE[info_hash]
            doc_id = info_hash

        if processed_data is not None:
            # if data loaded successfully from storage, just picking embeddings
            embeddings = []
            for chunk in processed_data:
                embeddings.append(chunk["embedding"])
            query_embedding = get_embeddings([data.query])[0]
        else:
            # data is new, need to parse and calculate embeddings
            # also need to save raw text to a designated collection in db
            row = {
                "document_id": doc_id,
                "text": data.text,
            }
            obj_id = DB[CONFIG["mongo"]["texts_collection"]].insert_one(row).inserted_id
            # todo: save doc_id -> obj_id mapping to a separate collection?

            text_chunks = parse_text(info, int(CONFIG["text_handler"]["chunk_size"]))
            text_chunks.append(data.query)
            embeddings = get_embeddings(text_chunks)
            assert len(embeddings) == len(text_chunks)
            query = text_chunks.pop()
            query_embedding = embeddings.pop()
            processed_data = []
            for chunk, emb in zip(text_chunks, embeddings):
                processed_data.append({"text": chunk, "embedding": emb})
            STORAGE[doc_id] = processed_data

        logging.info(
            f"Number of chunks of size {CONFIG['text_handler']['chunk_size']}: {len(processed_data)}"
        )
        cosines = [np.dot(emb, query_embedding) for emb in embeddings]
        indices = np.argsort(cosines)[-int(CONFIG["text_handler"]["top_k_chunks"]):][::-1]
        context = "\n\n".join([processed_data[i]["text"] for i in indices])
        logging.info(f"Top {CONFIG['text_handler']['top_k_chunks']} chunks:\n{context}")
    answer = get_answer(context, data.query)
    return answer, context, doc_id
