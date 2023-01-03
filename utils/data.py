from typing import Optional

from pydantic import BaseModel


class TextRequest(BaseModel):
    text_input: str
    query: str


class SetReactionRequest(BaseModel):
    request_id: str
    like: bool
    comment: Optional[str] = None
