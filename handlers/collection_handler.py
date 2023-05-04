import hashlib
import logging
import pickle
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from utils import CONFIG, DB, MILVUS_DB, ml_requests, hash_string
from utils.errors import (
    InvalidDocumentIdError,
)
from utils.schemas import CollectionQueryRequest, CollectionSolutionRequest


class CollectionHandler:
    def __init__(self, top_k_chunks: int, chunk_size: int):
        self.top_k_chunks = top_k_chunks
        self.chunk_size = chunk_size

    async def get_answer(
        self,
        request: CollectionQueryRequest,
        api_version: str,
    ) -> Tuple[str, str, List[str], List[str]]:
        collections = request.collections
        vendor = request.vendor
        org_hash = hash_string(request.organization_id)
        query_embedding = (await ml_requests.get_embeddings(request.query, api_version))[0]
        search_collections = [
            f"{vendor}_{org_hash}_{collection}" for collection in collections
        ]
        chunks, titles, doc_ids, doc_summaries = MILVUS_DB.search_collections_set(
            search_collections, query_embedding, self.top_k_chunks, api_version
        )
        context = "\n\n".join(chunks)

        answer = (
            await ml_requests.get_answer(
                context, request.query, api_version, "support", chat=request.chat
            )
        )["data"]

        return answer, context, doc_ids, titles, doc_summaries

    async def get_solution(
        self, request: CollectionSolutionRequest, api_version: str
    ) -> Tuple[str, str, List[str], List[str]]:
        collections = request.collections
        vendor = request.vendor
        org_hash = hash_string(request.organization_id)

        embedding, query = self.get_data_from_id(
            doc_id=request.document_id,
            full_collection_name=f"{vendor}_{org_hash}_{request.doc_collection}",
        )

        search_collections = [
            f"{vendor}_{org_hash}_{collection}" for collection in collections
        ]
        chunks, titles, doc_ids, doc_summaries = MILVUS_DB.search_collections_set(
            search_collections, embedding, self.top_k_chunks, api_version
        )
        context = "\n\n".join(chunks)

        answer = (await ml_requests.get_answer(context, query, api_version))["data"]

        return answer, context, doc_ids, titles, doc_summaries

    def get_data_from_id(self, doc_id: str, full_collection_name: str) -> np.ndarray:
        collection = MILVUS_DB[full_collection_name]
        res = collection.query(
            expr=f'doc_id=="{doc_id}"',
            offset=0,
            limit=30,
            output_fields=["chunk", "emb_v1"],
            consistency_level="Strong",
        )
        if len(res) != 1:
            col_name = full_collection_name.rsplit('_', maxsplit=1)[-1]
            if len(res) == 0:
                text = f"Unable to retrieve document with id {doc_id} in collection {col_name}"
            else:
                text = f"Ambiguous documents with id {doc_id} in collection {col_name}"
            raise InvalidDocumentIdError(text)
        emb = res[0]["emb_v1"]
        query = res[0]["chunk"]
        query += "\n\nAdress the problem stated above"

        return emb, query
