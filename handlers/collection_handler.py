import logging
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from utils import DB, ml_requests
from utils.schemas import ApiVersion, CollectionRequest


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
                if "doc_title" in chunks_embeddings[0] and "link" in chunks_embeddings[0]:
                    self.collections[api_version][collection][subcollection]["sources"] = [
                        (chunk["doc_title"], chunk["link"]) for chunk in chunks_embeddings
                    ]

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

    def get_answer(
        self,
        request: CollectionRequest,
        api_version: str,
    ) -> Tuple[str, str, Tuple[str, str] | None]:
        query_embedding = ml_requests.get_embeddings(request.query, api_version)[0]

        api_version_embeds = api_version if api_version in self.embeddings_sizes else "v1"

        subcollections = (
            request.subcollections
            if request.subcollections
            else self.collections[api_version_embeds][request.collection].keys()
        )

        chunks, embeddings, sources = (
            [],
            np.array([]).reshape(0, self.embeddings_sizes[api_version_embeds]),
            [],
        )
        for subcollection in subcollections:
            chunks.extend(
                self.collections[api_version_embeds][request.collection][subcollection]["chunks"]
            )
            embeddings = np.concatenate(
                (
                    embeddings,
                    self.collections[api_version_embeds][request.collection][subcollection][
                        "embeddings"
                    ],
                ),
                axis=0,
            )
            if (
                "sources"
                in self.collections[api_version_embeds][request.collection][subcollection]
            ):
                sources.extend(
                    self.collections[api_version_embeds][request.collection][subcollection][
                        "sources"
                    ]
                )

        context, indices = self.get_context_from_chunks_embeddings(
            chunks, embeddings, query_embedding
        )

        if sources:
            sources = [sources[i] for i in indices]
            sources = list(dict.fromkeys(sources))

        answer = ml_requests.get_answer(context, request.query, api_version, "support")

        return answer, context, sources

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
