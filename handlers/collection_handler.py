from collections import defaultdict
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from utils import CONFIG, DB, ml_requests
from utils.schemas import CollectionRequest
from utils import ml_requests


class CollectionHandler:
    def __init__(self, collections_prefix: str, top_k_chunks: int):
        self.top_k_chunks = top_k_chunks

        self.collections = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for name in DB.list_collection_names():
            if collections_prefix in name:
                api_version, _, collection, subcollection = name.split(".")
                self.collections[api_version][collection][subcollection] = DB[name].find({})

        print(self.collections)

    def get_answer(
        self,
        request: CollectionRequest,
        api_version: str,
    ) -> Tuple[str, str, str]:
        return

        chunks, embeddings = self.collections["api_version"][request.collection]

        context, indices = self.get_context_from_chunks_embeddings(
            chunks, embeddings, request.query, api_version
        )
        answer = ml_requests.get_answer(context, request.query, api_version)

        return answer, context

    def get_context_from_chunks_embeddings(
        self, chunks: List[str], embeddings: List[List[float]], query: str, api_version: str
    ) -> tuple[str, np.ndarray]:
        query_embedding = ml_requests.get_embeddings(query, api_version)[0]
        distances = [np.dot(embedding, query_embedding) for embedding in embeddings]
        indices = np.argsort(distances)[-int(self.top_k_chunks) :][::-1]
        context = "\n\n".join([chunks[i] for i in indices])
        return context, indices
