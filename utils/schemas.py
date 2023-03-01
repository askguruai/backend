from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ApiVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"


class TextRequest(BaseModel):
    text: str | None = Field(
        defalt=None,
        description="Text to generate answer from. Might be empty.",
        example="My name is Bob",
    )
    query: str = Field(description="Query to generate answer for.", example="What is your name?")


class LinkRequest(BaseModel):
    link: str = Field(
        description="Link to generate answer from.",
        example="https://en.wikipedia.org/wiki/2022_Russian_invasion_of_Ukraine",
    )
    query: str = Field(
        description="Query to generate answer for.",
        example="What are consequences of this invasion for Russia?",
    )


class DocumentRequest(BaseModel):
    document_id: str | List[str] = Field(
        description="Document ID(s) to generate answer from.",
        example="7af8c3e548e40aeb984c42dd",
    )
    query: str = Field(
        description="Query to generate answer for.",
        example="What are consequences of this invasion for Russia?",
    )


class GetAnswerResponse(BaseModel):
    answer: str = Field(
        description="Answer to the query",
        example="I used to play drums.",
    )
    request_id: str = Field(
        description="A request id which is used to /set_reaction.",
        example="63cbd74e8d31a62a1512eab1",
    )
    info_source: List[Dict] | None = Field(
        default=None,
        description="A list of dictionaries with information about the source of the answer.",
    )


class UploadDocumentResponse(BaseModel):
    document_id: str = Field(
        description="ID of an uploaded document.", example="7af8c3e548e40aeb984c42dd"
    )


class LikeStatus(str, Enum):
    wrong_answer = "wrong_answer"
    incomplete_answer = "incomplete_answer"
    offensive_answer = "offensive_answer"
    good_answer = "good_answer"


class SetReactionRequest(BaseModel):
    request_id: str = Field(
        description="Request ID to set reaction for.",
        example="63cbd74e8d31a62a1512eab1",
    )
    like_status: LikeStatus = Field(description="Reaction to set.", example=LikeStatus.good_answer)
    comment: str | None = Field(
        default=None, description="Comment to set.", example="Very accurate!"
    )

    class Config:
        use_enum_values = True


class HTTPExceptionResponse(BaseModel):
    detail: str = Field(example="Internal Server Error")
