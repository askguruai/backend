from enum import Enum
from typing import Dict, List, Tuple

from fastapi import status
from pydantic import BaseModel, Field


class ApiVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"
    v3 = "v3"


class HTTPExceptionResponse(BaseModel):
    detail: str = Field(example="Internal Server Error")


class AuthExceptionResponse(BaseModel):
    detail: str = Field(example="Could not validate credentials")


collection_responses = {
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse},
    status.HTTP_401_UNAUTHORIZED: {"model": AuthExceptionResponse},
}

AUTH_METHODS = ["org_scope", "default"]


class GetCollectionResponse(BaseModel):
    doc_ids: List[str] = Field(
        description="List of doc ids from given collection",
        example=["7af8c3e548e40aeb984c42dd", "7af8c3e548e40aeb984c42de"],
    )


class AuthenticatedRequest(BaseModel):
    auth_method: str = Field(
        description=f"Auth strategy name. Possible values: {', '.join(AUTH_METHODS)}",
        example="default",
    )


class VendorCollectionRequest(BaseModel):
    vendor: str = Field(description="Vendor that hosts data", example="livechat")
    organization: str = Field(
        description=f"aka collection to use",
        example="f1ac8408-27b2-465e-89c6-b8708bfc262c",
    )


class VendorCollectionTokenRequest(VendorCollectionRequest):
    password: str = Field(description="This is for staff use")


class Source(BaseModel):
    id: str = Field(description="Id of the source", example="123456")
    title: str = Field(description="Title of the source", example="Payment")
    collection: str = Field(description="Collection of the source", example="internal")
    summary: str | None = Field(
        description="Summary of the source",
        example="Payment methods and informaton summary. How to pay for subscription",
    )


class GetCollectionRankingResponse(BaseModel):
    sources: List[Source] = Field(
        description="List of sources from given collection",
        example=[
            Source(
                id="123456",
                title="Payment",
                collection="internal",
                summary="Payment methods and informaton summary. How to pay for subscription",
            )
        ],
    )


class CollectionQueryRequest(VendorCollectionRequest):
    query: str = Field(description="Query to generate answer for.", example="What is your name?")
    collections: List[str] = Field(
        description=f"Collections to use.",
        example=["chats", "tickets"],
    )
    chat: List[dict] | None = Field(
        description="Optional: ongoing chat with the client",
        example=[
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "do you offer screen sharing chat"},
            {
                "role": "assistant",
                "content": "Hello, I will check, thanks for waiting.",
            },
            {"role": "user", "content": "Sure."},
        ],
    )
    n_top_ranking: int | None = Field(
        description="Number of most relevant sources to be returned",
        default=3,
        example=3,
    )


class CollectionSolutionRequest(VendorCollectionRequest):
    document_id: str = Field(
        description="Doc id. Use only if you know what this is.", default=None
    )
    doc_collection: str = Field(
        description="Collection where to look for document id",
        default=None,
    )
    collections: List[str] = Field(
        description=f"Collections to use.",
        example=["chats", "tickets"],
    )
    n_top_ranking: int | None = Field(
        description="Number of most relevant sources to be returned",
        default=3,
        example=3,
    )


class LivechatLoginRequest(BaseModel):
    livechat_token: str = Field(
        description="Token received through livechat auth",
    )


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


class UploadChatsRequest(VendorCollectionRequest):
    chats: List[dict] = Field(
        description="A list of all archive chat objects",
        example=[
            {
                "id": "RSF0JQSFEJ",
                "user": {"name": "Mike", "id": "c5753edc-0051-4a7a-8b61-2ae64c7aad51"},
                "history": [
                    {"author": "agent", "text": "Hello. How may I help you?"},
                    {"author": "user", "text": "hi do you offer screen sharing chat"},
                ],
            },
            {
                "id": "RSF0JQB8J4",
                "user": {"name": "Bob", "id": "efb1f05f-dd19-4b95-95e9-3cd2feb149bf"},
                "history": [
                    {"author": "agent", "text": "Hello. How may I help you?"},
                    {
                        "author": "user",
                        "text": "i want to change my subscription from monthly to annually",
                    },
                ],
            },
        ],
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
    sources: List[Source] | None = Field(
        default=None,
        description="A list of dictionaries with information about the source of the answer.",
    )


class GetAnswerCollectionResponse(BaseModel):
    answer: str = Field(
        description="Answer to the query",
        example="I used to play drums.",
    )
    request_id: str | None = Field(
        description="A request id which is used to /set_reaction.",
        example="63cbd74e8d31a62a1512eab1",
    )
    sources: List[Source] | None = Field(
        default=None,
        description="A list of dictionaries with information about the source of the answer.",
    )


class UploadDocumentResponse(BaseModel):
    document_id: str = Field(
        description="ID of an uploaded document.", example="7af8c3e548e40aeb984c42dd"
    )


class UploadChatsResponse(BaseModel):
    uploaded_chunks_number: str = Field(
        description="Number of chunks successfully uploaded", example="5"
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
