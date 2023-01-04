from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TextRequest(BaseModel):
    text_input: str
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
