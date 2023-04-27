import logging
import pickle
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from utils import DB, ml_requests
from utils.errors import (
    InvalidDocumentIdError,
    RequestDataModelMismatchError,
    SubcollectionDoesNotExist,
)
from utils.schemas import CollectionRequest


class CollectionHandler:
    def __init__(self, collections_prefix: str, top_k_chunks: int, chunk_size: int):
        self.top_k_chunks = top_k_chunks
        self.chunk_size = chunk_size

        self.collections = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )

        for name in DB.list_collection_names():
            if collections_prefix in name:
                structure = name.split(".")
                if len(structure) == 5:
                    api_version, _, vendor, collection, subcollection = name.split(".")
                    chunks_embeddings = list(DB[name].find({}))
                    self.collections[api_version][vendor][collection][subcollection]["chunks"] = [
                        chunk["chunk"] for chunk in chunks_embeddings
                    ]
                    self.collections[api_version][vendor][collection][subcollection][
                        "embeddings"
                    ] = [pickle.loads(chunk["embedding"]) for chunk in chunks_embeddings]
                    if "doc_title" in chunks_embeddings[0] and "link" in chunks_embeddings[0]:
                        if "description" in chunks_embeddings[0]:
                            self.collections[api_version][vendor][collection][subcollection][
                                "sources"
                            ] = [
                                [chunk["doc_title"], chunk["link"], chunk["description"]]
                                for chunk in chunks_embeddings
                            ]
                        else:
                            self.collections[api_version][vendor][collection][subcollection][
                                "sources"
                            ] = [
                                [chunk["doc_title"], chunk["link"]] for chunk in chunks_embeddings
                            ]

        logs = CollectionHandler.get_dict_logs(self.collections)
        logging.info(logs)

        self.embeddings_sizes = {'v1': 1536, 'v2': 768}  # idk maybe make it proper way later
        # this ugly piece of crap just loads random
        # embedding for each api version and remembers
        # its size
        # for api_version in self.collections:
        #     self.embeddings_sizes[api_version] = self.collections[api_version].values().
        #
        #
        #     self.embeddings_sizes[api_version] = self.collections[api_version][
        #         list(self.collections[api_version].keys())[0]
        #     ][
        #         list(
        #             self.collections[api_version][
        #                 list(self.collections[api_version].keys())[0]
        #             ].keys()
        #         )[0]
        #     ][
        #         "embeddings"
        #     ].shape[
        #         1
        #     ]
        # logging.info(f"Embedding sizes:\n{self.embeddings_sizes}")

    async def get_answer(
        self,
        request: CollectionRequest,
        api_version: str,
    ) -> Tuple[str, str, List[str] | None]:
        api_version_embeds = api_version if api_version in self.embeddings_sizes else "v1"

        subcollections = (
            request.subcollections
            if request.subcollections
            else self.collections[api_version_embeds][request.organization_id].keys()
        )

        query = (
            request.query
            if request.query
            else self.get_query_from_id(
                doc_id=request.document_id,
                org_id=request.organization_id,
                subcollections=["tickets"]
                if request.organization_id == "vivantio"
                and request.subcollections == ["internal"]
                and request.document_id
                else subcollections,
                api_ver=api_version_embeds,
                vendor=request.vendor,
            )
        )

        query_embedding = (await ml_requests.get_embeddings(query, api_version))[0]

        chunks, embeddings, sources = (
            [],
            np.array([]).reshape(0, self.embeddings_sizes[api_version_embeds]),
            [],
        )
        for subcollection in subcollections:
            if (
                "chunks"
                not in self.collections[api_version_embeds][request.vendor][
                    request.organization_id
                ][subcollection]
            ):
                raise SubcollectionDoesNotExist()
            chunks.extend(
                self.collections[api_version_embeds][request.vendor][request.organization_id][
                    subcollection
                ]["chunks"]
            )
            embeddings = np.concatenate(
                (
                    embeddings,
                    np.array(
                        self.collections[api_version_embeds][request.vendor][
                            request.organization_id
                        ][subcollection]["embeddings"]
                    ),
                ),
                axis=0,
            )
            if (
                "sources"
                in self.collections[api_version_embeds][request.vendor][request.organization_id][
                    subcollection
                ]
            ):
                sources.extend(
                    self.collections[api_version_embeds][request.vendor][request.organization_id][
                        subcollection
                    ]["sources"]
                )

        context, indices = self.get_context_from_chunks_embeddings(
            chunks, embeddings, query_embedding, return_top_k=request.n_top_ranking
        )

        if sources:
            sources = [sources[i] for i in indices]
            # sources = list(dict.fromkeys(sources))

        if request.organization_id == "vivantio" and (
            request.subcollections == ["tickets"]
            or (request.subcollections == ["internal"] and request.n_top_ranking != 0)
        ):
            answer = ""
        else:
            answer = (
                await ml_requests.get_answer(
                    context, query, api_version, "support", chat=request.chat
                )
            )["data"]

        return answer, context, sources

    def get_query_from_id(
        self, doc_id: str, org_id: str, subcollections: List[str], api_ver: str, vendor: str
    ) -> str:
        document = None
        for sub in subcollections:
            document = DB[f"{api_ver}.collections.{vendor}.{org_id}.{sub}"].find_one(
                {"doc_id": doc_id}
            )
            if document is not None:
                print("Document found!")
                chunk = document["chunk"]
                # removing solution so not to mislead model
                start, end = chunk.rsplit("\n", maxsplit=1)
                if end.startswith("Solution:"):
                    chunk = start
                return chunk
        raise InvalidDocumentIdError(f"Requested document with id {doc_id} was not found")

    def get_context_from_chunks_embeddings(
        self,
        chunks: List[str],
        embeddings: NDArray,
        query_embedding: np.ndarray,
        return_top_k: int = None,
    ) -> tuple[str, np.ndarray]:
        similarities = np.dot(embeddings, query_embedding)
        take_top_k = self.top_k_chunks
        if return_top_k is None:
            return_top_k = take_top_k
        slice = max(take_top_k, return_top_k)
        indices = np.argsort(similarities)[-slice:][::-1]
        context = "\n\n".join([chunks[i] for i in indices[:take_top_k]])
        context = context[: self.chunk_size * take_top_k]
        return context, indices[:return_top_k]

    @staticmethod
    def get_dict_logs(d, indent=0, logs='Collections structure:\n'):
        for key, value in sorted(d.items()):
            if isinstance(value, dict):
                logs += '  ' * indent + f"{key}: \n"
                logs = CollectionHandler.get_dict_logs(value, indent + 1, logs)
            else:
                logs += '  ' * indent + f"{key}: {len(value)}\n"
        return logs
