import abc
import hashlib
from typing import Any, Dict, List, Union

from bson.objectid import ObjectId

from db import DB
from utils import STORAGE, ml


class GeneralHandler:
    def __init__(self):
        pass

    def get_answer(
        self, request: Union[TextRequest, LinkRequest, DocRequest]
    ) -> Tuple[str, str, str]:
        text = self.get_text_from_request(request)
        if not text:
            return ml.get_answer("", request.query), "", ""

        text_hash = self.get_hash(text)

        document = DB[CONFIG["mongo"]["requests_inputs_collection"]].find_one(
            {"_id": ObjectId(text_hash)}
        )
        if not document:
            chunks = self.get_chunks_from_text(text)
            embeddings = self.get_embeddings_from_chunks(chunks)
            document = {
                "_id": ObjectId(text_hash),
                "text": text,
                "chunks": chunks,
                "embeddings": embeddings,
            } | self.get_additional_request_data(request)
            DB[CONFIG["mongo"]["requests_inputs_collection"]].insert_one(document)

        chunks, embeddings = document["chunks"], document["embeddings"]

        context = self.get_context_from_chunks_embeddings(chunks, embeddings, query)
        answer = ml.get_answer(context, data.query)

        return answer, context, text_hash

    def get_chunks_embeddings(self, text: str) -> Tuple[List[str], List[List[float]]]:
        sentences = text_to_sentences(raw_text)
        chunks = chunkise_sentences(sentences, chunk_size)
        pass

    def get_chunks_from_text(self, text: str) -> List[str]:
        pass

    def get_embeddings_from_chunks(self, chunks: List[str]) -> List[List[float]]:
        pass

    def get_context_from_chunks_embeddings(
        self, chunks: List[str], embeddings: List[List[float]], uery: str
    ) -> str:
        pass

    @staticmethod
    def get_hash(self, text: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    @abc.abstractmethod
    def get_additional_request_data(
        self, request: Union[TextRequest, LinkRequest, DocRequest]
    ) -> Dict[str, Any]:
        return

    @abc.abstractmethod
    def get_text_from_request(self, request: Union[TextRequest, LinkRequest, DocRequest]) -> str:
        return
