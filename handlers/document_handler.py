from typing import Any, Dict, Tuple

from handlers.general_handler import GeneralHandler
from utils.api import DocumentRequest
from utils import DB, CONFIG, ml_requests
from bson.objectid import ObjectId
from utils.errors import InvalidDocumentIdError
import pickle


class DocumentHandler(GeneralHandler):
    def get_answer(
        self, request: DocumentRequest
    ) -> Tuple[str, str, str]:
        document_id = request.document_id
        document = DB[CONFIG["mongo"]["requests_inputs_collection"]].find_one(
            {"_id": ObjectId(document_id)}
        )
        if document is None:
            raise InvalidDocumentIdError()
        chunks, embeddings = document["chunks"], pickle.loads(document["embeddings"])
        context = self.get_context_from_chunks_embeddings(chunks, embeddings, request.query)
        answer = ml_requests.get_answer(context, request.query)
        return answer, context, document_id

    def get_additional_request_data(self, request: DocumentRequest) -> Dict[str, Any]:
        return {"document_id": request.document_id}

    def get_text_from_request(self, request: DocumentRequest) -> str:
        return self.parser.get_text(request.document_id)
