from typing import Any, Dict

from handlers.general_handler import GeneralHandler
from utils.api import TextRequest


class TextHandler(GeneralHandler):
    def get_additional_request_data(self, request: TextRequest) -> Dict[str, Any]:
        return {}

    def get_text_from_request(self, request: TextRequest) -> str:
        return self.parser.get_text(request.text)
