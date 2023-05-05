import hashlib
import logging
import pickle
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from utils import CONFIG, DB, MILVUS_DB, hash_string, ml_requests
from utils.errors import InvalidDocumentIdError
from utils.schemas import (
    ApiVersion,
    CollectionSolutionRequest,
    GetCollectionAnswerResponse,
    GetCollectionRankingResponse,
    GetCollectionResponse,
    Source,
)


class CollectionHandler:
    def __init__(self, top_k_chunks: int, chunk_size: int):
        self.top_k_chunks = top_k_chunks
        self.chunk_size = chunk_size

    async def get_answer(
        self,
        vendor: str,
        organization: str,
        collections: List[str],
        query: str,
        api_version: ApiVersion,
    ) -> GetCollectionAnswerResponse:
        org_hash = hash_string(organization)
        query_embedding = (await ml_requests.get_embeddings(query, api_version.value))[0]
        search_collections = [f"{vendor}_{org_hash}_{collection}" for collection in collections]
        chunks, titles, doc_ids, doc_summaries, doc_collections = MILVUS_DB.search_collections_set(
            search_collections, query_embedding, self.top_k_chunks, api_version
        )
        context = "\n\n".join(chunks)

        answer = await ml_requests.get_answer(context, query, api_version.value, "support")

        return GetCollectionAnswerResponse(
            answer=answer,
            sources=[
                Source(
                    id=doc_id,
                    title=title,
                    collection=collection.split("_")[-1],
                    summary=doc_summary,
                )
                for title, doc_id, doc_summary, collection in zip(
                    titles, doc_ids, doc_summaries, doc_collections
                )
            ],
        )

    async def get_solution(
        self,
        vendor: str,
        organization: str,
        collections: List[str],
        document: str,
        document_collection: str,
        api_version: ApiVersion,
    ) -> GetCollectionAnswerResponse:
        org_hash = hash_string(organization)

        embedding, query = self.get_data_from_id(
            document=document,
            full_collection_name=f"{vendor}_{org_hash}_{document_collection}",
        )

        search_collections = [f"{vendor}_{org_hash}_{collection}" for collection in collections]
        chunks, titles, doc_ids, doc_summaries, doc_collections = MILVUS_DB.search_collections_set(
            search_collections, embedding, self.top_k_chunks, api_version
        )
        context = "\n\n".join(chunks)

        answer = await ml_requests.get_answer(context, query, api_version)

        return GetCollectionAnswerResponse(
            answer=answer,
            sources=[
                Source(
                    id=doc_id,
                    title=title,
                    collection=collection.split("_")[-1],
                    summary=doc_summary,
                )
                for title, doc_id, doc_summary, collection in zip(
                    titles, doc_ids, doc_summaries, doc_collections
                )
            ],
        )

    def get_data_from_id(self, document: str, full_collection_name: str) -> np.ndarray:
        collection = MILVUS_DB[full_collection_name]
        res = collection.query(
            expr=f'doc_id=="{document}"',
            offset=0,
            limit=30,
            output_fields=["chunk", "emb_v1"],
            consistency_level="Strong",
        )
        if len(res) != 1:
            col_name = full_collection_name.rsplit("_", maxsplit=1)[-1]
            if len(res) == 0:
                text = f"Unable to retrieve document {document} in collection {col_name}"
            else:
                text = f"Ambiguous documents {document} in collection {col_name}"
            raise InvalidDocumentIdError(text)
        emb = res[0]["emb_v1"]
        query = res[0]["chunk"]
        query += "\n\nAdress the problem stated above"

        return emb, query

    def get_collection(
        self, vendor: str, organization: str, collection: str, api_version: ApiVersion
    ) -> GetCollectionResponse:
        organization_hash = hash_string(organization)
        full_collection_name = f"{vendor}_{organization_hash}_{collection}"
        milvus_collection = MILVUS_DB[full_collection_name]
        # TODO this might not retrive all docs
        # if will have int pk then retrieve by it
        chunks = milvus_collection.query(
            expr='doc_id >= "0"',
            output_fields=["doc_id"],  # TODO add ts later as well
        )
        return GetCollectionResponse(doc_ids=list(set([chunk["doc_id"] for chunk in chunks])))

    async def get_ranking(
        self,
        vendor: str,
        organization: str,
        top_k: int,
        api_version: ApiVersion,
        query: str = None,
        document: str = None,
        document_collection: str = None,
        collections: List[str] = None,
    ) -> GetCollectionRankingResponse:
        organization_hash = hash_string(organization)
        if query:
            embedding = (await ml_requests.get_embeddings(query, api_version.value))[0]
        elif document:
            document_collection = document_collection or "faq"
            embedding, _ = self.get_data_from_id(
                document=document,
                full_collection_name=f"{vendor}_{organization_hash}_{document_collection}",
            )

        collections_search = [
            f"{vendor}_{organization_hash}_{collection}" for collection in collections
        ]
        _, titles, doc_ids, doc_summaries, doc_collections = MILVUS_DB.search_collections_set(
            collections_search, embedding, top_k, api_version.value
        )

        return GetCollectionRankingResponse(
            sources=[
                Source(
                    id=doc_id,
                    title=title,
                    collection=collection.split("_")[-1],
                    summary=doc_summary,
                )
                for title, doc_id, doc_summary, collection in zip(
                    titles, doc_ids, doc_summaries, doc_collections
                )
            ]
        )
