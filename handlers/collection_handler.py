import hashlib
import logging
import pickle
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from utils import CONFIG, DB, MILVUS_DB, ml_requests
from utils.errors import (
    InvalidDocumentIdError,
    RequestDataModelMismatchError,
    SubcollectionDoesNotExist,
)
from utils.schemas import CollectionRequest


class CollectionHandler:
    def __init__(self, top_k_chunks: int, chunk_size: int):
        self.top_k_chunks = top_k_chunks
        self.chunk_size = chunk_size

    async def get_answer(
        self,
        request: CollectionRequest,
        api_version: str,
    ) -> Tuple[str, str, List[str] | None]:
        query_embedding = (await ml_requests.get_embeddings(request.query, api_version))[0]

        subcollections = request.subcollections
        vendor = request.vendor
        org_id = request.organization_id
        org_hash = hashlib.sha256(org_id.encode()).hexdigest()[: int(CONFIG["misc"]["hash_size"])]

        search_collections = [
            f"{vendor}_{org_hash}_{subcollection}" for subcollection in subcollections
        ]
        chunks, titles = MILVUS_DB.search_collections_set(
            search_collections, query_embedding, self.top_k_chunks, api_version
        )
        context = "\n\n".join(chunks)

        answer = (
            await ml_requests.get_answer(
                context, request.query, api_version, "support", chat=request.chat
            )
        )["data"]

        return answer, context, titles
