import abc
import hashlib
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from bson.binary import Binary
from bson.objectid import ObjectId

from parsers import DocumentParser, LinkParser, TextParser
from utils import CONFIG, DB, ml_requests
from utils.api import DocumentRequest, LinkRequest, TextRequest


class GeneralHandler:
    def __init__(self, parser: Union[TextParser, LinkParser, DocumentParser], top_k_chunks: int):
        self.parser = parser
        self.top_k_chunks = top_k_chunks

    def get_answer(
        self, request: Union[TextRequest, LinkRequest, DocumentRequest]
    ) -> Tuple[str, str, str]:
        text = self.get_text_from_request(request)
        if not text:
            return ml_requests.get_answer("", request.query), "", ""

        text_hash = GeneralHandler.get_hash(text)

        document = DB[CONFIG["mongo"]["requests_inputs_collection"]].find_one(
            {"_id": ObjectId(text_hash)}
        )
        if not document:
            chunks = self.parser.get_chunks_from_text(text)
            embeddings = self.get_embeddings_from_chunks(chunks)
            document = {
                "_id": ObjectId(text_hash),
                "text": text,
                "chunks": chunks,
                "embeddings": Binary(pickle.dumps(embeddings)),
            } | self.get_additional_request_data(request)
            DB[CONFIG["mongo"]["requests_inputs_collection"]].insert_one(document)
        else:
            chunks, embeddings = document["chunks"], pickle.loads(document["embeddings"])

        context, indices = self.get_context_from_chunks_embeddings(chunks, embeddings, request.query)
        answer = ml_requests.get_answer(context, request.query)

        return answer, context, text_hash

    def get_embeddings_from_chunks(self, chunks: List[str]) -> List[List[float]]:
        embeddings = ml_requests.get_embeddings(chunks)
        assert len(embeddings) == len(chunks)
        return embeddings

    def get_context_from_chunks_embeddings(
        self, chunks: List[str], embeddings: List[List[float]], query: str
    ) -> tuple[str, np.ndarray]:
        query_embedding = ml_requests.get_embeddings(query)[0]
        distances = [np.dot(embedding, query_embedding) for embedding in embeddings]
        indices = np.argsort(distances)[-int(self.top_k_chunks) :][::-1]
        context = "\n\n".join([chunks[i] for i in indices])
        return context, indices

    @staticmethod
    def get_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    @abc.abstractmethod
    def get_additional_request_data(
        self, request: Union[TextRequest, LinkRequest, DocumentRequest]
    ) -> Dict[str, Any]:
        return

    @abc.abstractmethod
    def get_text_from_request(
        self, request: Union[TextRequest, LinkRequest, DocumentRequest]
    ) -> str:
        return
