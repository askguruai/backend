import logging
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from utils import DB, ml_requests
from utils.schemas import ApiVersion, CollectionRequest, ResponseSourceArticle, ResponseSourceChat
from milvus_db.utils import get_collection, search_collections_set
from pymilvus import Collection
import hashlib


class CollectionHandler:
    def __init__(self, top_k_chunks: int, chunk_size: int):
        self.top_k_chunks = top_k_chunks
        self.chunk_size = chunk_size

        

    def get_answer(
        self,
        request: CollectionRequest,
        api_version: str,
    ) -> Tuple[str, str, List[str] | None]:
        query_embedding = ml_requests.get_embeddings(request.query, api_version)[0]

        subcollections = request.subcollections
        vendor = request.vendor
        org_id = request.organization_id
        org_hash = hashlib.sha256(org_id.encode()).hexdigest()[:24]

        search_collections = [
            get_collection(f"{vendor}_{org_hash}_{subcollection}") for subcollection in subcollections
        ]

        chunks, titles = search_collections_set(search_collections, query_embedding, self.top_k_chunks, api_version)
        context = "\n\n".join(chunks)

        answer = ml_requests.get_answer(
            context, request.query, api_version, "support", chat=request.chat
        )

        return answer, context, titles 

    def get_context_from_chunks_embeddings(
        self, chunks: List[str], embeddings: NDArray, query_embedding: np.ndarray
    ) -> tuple[str, np.ndarray]:
        distances = np.dot(embeddings, query_embedding)
        indices = np.argsort(distances)[-int(self.top_k_chunks) :][::-1]
        context = "\n\n".join([chunks[i] for i in indices])
        context = context[: self.chunk_size * self.top_k_chunks]
        return context, indices

    @staticmethod
    def get_dict_logs(d, indent=0, logs='Collections structure:\n'):
        for key, value in sorted(d.items()):
            if isinstance(value, dict):
                logs += '  ' * indent + f"{key}: \n"
                logs = CollectionHandler.get_dict_logs(value, indent + 1, logs)
            else:
                logs += '  ' * indent + f"{key}\n"
        return logs
