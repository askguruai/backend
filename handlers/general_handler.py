import abc
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from bson.binary import Binary
from bson.objectid import ObjectId

from parsers import DocumentParser, LinkParser, TextParser
from utils import CONFIG, DB, hash_string, ml_requests
from utils.schemas import DocumentRequest, LinkRequest, TextRequest


class GeneralHandler:
    def __init__(self, parser: Union[TextParser, LinkParser, DocumentParser], top_k_chunks: int):
        self.parser = parser
        self.top_k_chunks = top_k_chunks

    async def get_answer(
        self,
        request: Union[TextRequest, LinkRequest, DocumentRequest],
        api_version: str,
    ) -> Tuple[str, str, str]:
        text = self.get_text_from_request(request)
        if not text:
            return await ml_requests.get_answer("", request.query, api_version), "", ""

        text_hash = hash_string(text)

        document = DB[api_version + CONFIG["mongo"]["requests_inputs_collection"]].find_one(
            {"_id": ObjectId(text_hash)}
        )
        if not document:
            chunks = self.parser.get_chunks_from_text(text)
            embeddings = await self.get_embeddings_from_chunks(chunks, api_version)
            document = {
                "_id": ObjectId(text_hash),
                "text": text,
                "chunks": chunks,
                "embeddings": Binary(pickle.dumps(embeddings)),
            } | self.get_additional_request_data(request)
            DB[api_version + CONFIG["mongo"]["requests_inputs_collection"]].insert_one(document)
        else:
            chunks, embeddings = document["chunks"], pickle.loads(document["embeddings"])

        context, indices = await self.get_context_from_chunks_embeddings(chunks, embeddings, request.query, api_version)
        answer = await ml_requests.get_answer(context, request.query, api_version)

        return answer, context, text_hash

    async def get_embeddings_from_chunks(self, chunks: List[str], api_version: str) -> List[List[float]]:
        embeddings = await ml_requests.get_embeddings(chunks, api_version)
        assert len(embeddings) == len(chunks)
        return embeddings

    async def get_context_from_chunks_embeddings(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        query: str,
        api_version: str,
    ) -> tuple[str, np.ndarray]:
        query_embedding = (await ml_requests.get_embeddings(query, api_version))[0]
        similarities = [np.dot(embedding, query_embedding) for embedding in embeddings]
        indices = np.argsort(similarities)[-int(self.top_k_chunks) :][::-1]
        context = "\n\n".join([chunks[i] for i in indices])
        return context, indices

    @abc.abstractmethod
    def get_additional_request_data(self, request: Union[TextRequest, LinkRequest, DocumentRequest]) -> Dict[str, Any]:
        return

    @abc.abstractmethod
    def get_text_from_request(self, request: Union[TextRequest, LinkRequest, DocumentRequest]) -> str:
        return
