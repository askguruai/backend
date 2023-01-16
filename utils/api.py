from enum import Enum
from typing import Optional, Union, List

from pydantic import BaseModel


class TextRequest(BaseModel):
    text: Optional[str] = None
    query: str


class LinkRequest(BaseModel):
    link: Optional[str] = None
    query: str


class DocumentRequest(BaseModel):
    document_id: Union[str, List[str]] = None
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
