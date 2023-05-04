import logging
import pickle
from typing import Any, Dict, List, Tuple

from bson.objectid import ObjectId

from handlers.general_handler import GeneralHandler
from utils import CONFIG, DB, ml_requests
from utils.errors import InvalidDocumentIdError, RequestDataModelMismatchError
from utils.schemas import DocumentRequest


class DocumentHandler(GeneralHandler):
    async def get_answer(
        self, request: DocumentRequest, api_version: str
    ) -> Tuple[str, str, List[Dict], List[str]]:
        if isinstance(request.document_id, str):
            document_ids = [request.document_id]
        elif isinstance(request.document_id, list):
            document_ids = request.document_id
        else:
            raise RequestDataModelMismatchError(
                f"document_id param should be either str or list, "
                f"but is {type(request.document_id)}"
            )
        all_chunks, all_embeddings, doc_ids = (
            [],
            [],
            [],
        )  # todo: consider cases with huge doc input. will we run into oom?
        for doc_id in document_ids:
            document = DB[api_version + CONFIG["mongo"]["requests_inputs_collection"]].find_one(
                {"_id": ObjectId(doc_id)}
            )
            if document is None:
                # or just skip that doc? to think about it
                raise InvalidDocumentIdError(f"invalid document_id: {doc_id}")
            logging.info(f"Document {doc_id} is found in the database")
            chunks, embeddings = document["chunks"], pickle.loads(document["embeddings"])
            all_embeddings.extend(embeddings)
            all_chunks.extend(chunks)
            doc_ids.extend([doc_id] * len(chunks))
        context, indices = await self.get_context_from_chunks_embeddings(
            all_chunks, all_embeddings, request.query, api_version
        )
        info_sources = [
            {"document": doc_ids[global_idx], "chunk": all_chunks[global_idx]}
            for global_idx in indices
        ]
        answer = await ml_requests.get_answer(context, request.query, api_version)
        return answer, context, info_sources, document_ids

    def get_additional_request_data(self, request: DocumentRequest) -> Dict[str, Any]:
        return {"document_id": request.document_id}

    def get_text_from_request(self, request: DocumentRequest) -> str:
        return self.parser.get_text(request.document_id)
