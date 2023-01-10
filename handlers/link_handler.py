from typing import Any, Dict

from handlers.general_handler import GeneralHandler
from utils.api import LinkRequest


class LinkHandler(GeneralHandler):
    def get_additional_request_data(self, request: LinkRequest) -> Dict[str, Any]:
        return {"link": request.link}

    def get_text_from_request(self, request: LinkRequest) -> str:
        return self.parser.get_text(request.link)
