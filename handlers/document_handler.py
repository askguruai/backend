from typing import Any, Dict

from handlers.general_handler import GeneralHandler
from utils.api import DocumentRequest


class DocumentHandler(GeneralHandler):
    def get_additional_request_data(self, request: DocumentRequest) -> Dict[str, Any]:
        return {"document_id": request.document_id}

    def get_text_from_request(self, request: DocumentRequest) -> str:
        return self.parser.get_text(request.document_id)
