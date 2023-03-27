import logging
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from utils import DB, ml_requests
from utils.schemas import CollectionRequest


class CollectionHandler:
    def __init__(self, collections_prefix: str, top_k_chunks: int, chunk_size: int):
        self.top_k_chunks = top_k_chunks
        self.chunk_size = chunk_size

        self.collections = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for name in DB.list_collection_names():
            if collections_prefix in name:
                api_version, _, collection, subcollection = name.split(".")
                chunks_embeddings = list(DB[name].find({}))
                self.collections[api_version][collection][subcollection]["chunks"] = [
                    chunk["chunk"] for chunk in chunks_embeddings
                ]
                self.collections[api_version][collection][subcollection]["embeddings"] = np.array(
                    [pickle.loads(chunk["embedding"]) for chunk in chunks_embeddings]
                )

        logs = CollectionHandler.get_dict_logs(self.collections)
        logging.info(logs)

        # this ugly piece of crap just loads random
        # embedding for each api version and remembers
        # its size
        self.embeddings_sizes = {}
        for api_version in self.collections:
            self.embeddings_sizes[api_version] = self.collections[api_version][
                list(self.collections[api_version].keys())[0]
            ][
                list(
                    self.collections[api_version][
                        list(self.collections[api_version].keys())[0]
                    ].keys()
                )[0]
            ][
                "embeddings"
            ].shape[
                1
            ]
        logging.info(f"Embedding sizes:\n{self.embeddings_sizes}")

        # logging.info(len(self.collections["v2"]["livechat"]["chatbot"]["embeddings"][0]))

    def get_answer(
        self,
        request: CollectionRequest,
        api_version: str,
    ) -> Tuple[str, str, str]:
        subcollections = (
            request.subcollections
            if request.subcollections
            else self.collections[api_version][request.collection].keys()
        )

        chunks, embeddings = [], np.array([]).reshape(0, self.embeddings_sizes[api_version])
        for subcollection in subcollections:
            chunks.extend(
                self.collections[api_version][request.collection][subcollection]["chunks"]
            )
            embeddings = np.concatenate(
                (
                    embeddings,
                    self.collections[api_version][request.collection][subcollection]["embeddings"],
                ),
                axis=0,
            )

        context, indices = self.get_context_from_chunks_embeddings(
            chunks, embeddings, request.query, api_version
        )

        answer = ml_requests.get_answer(context, request.query, api_version, "support")

        return answer, context

    def get_context_from_chunks_embeddings(
        self, chunks: List[str], embeddings: NDArray, query: str, api_version: str
    ) -> tuple[str, np.ndarray]:
        query_embedding = ml_requests.get_embeddings(query, api_version)[0]
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
