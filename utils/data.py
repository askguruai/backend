from enum import Enum
from typing import Optional

from pydantic import BaseModel


class PdfQueryRequest(BaseModel):
    document_id: str
    query: str


class TextQueryRequest(BaseModel):
    text: Optional[str] = None
    query: str
    document_id: Optional[str] = None


class QueryRequest(BaseModel):
    text: Optional[str] = None
    link: Optional[str] = None
    doc: Optional[str] = None
    query: str


class LikeStatus(str, Enum):
    wrong_answer = "wrong_answer"
    incomplete_answer = "incomplete_answer"
    offensive_answer = "offensive_answer"
    good_answer = "good_answer"


class SetReactionRequest(BaseModel):
    request_id: str
    like_status: LikeStatus
    comment: Optional[str] = None

    class Config:
        use_enum_values = True
