from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, root_validator


class ApiVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"
    v3 = "v3"


class Collection(str, Enum):
    livechat = "livechat"
    groovehq = "groovehq"
    vivantio = "vivantio"
    askguru = "askguru"


SubCollections = {
    "livechat": ["chatbot", "helpdesk", "livechat", "knowledgebase", "internal"],
    "groovehq": ["public"],
    "vivantio": ["internal"],
    "askguru": ["chats"],
}

AUTH_METHODS = ["org_scope", "default"]


class AuthenticatedRequest(BaseModel):
    auth_method: str = Field(
        description=f"Auth strategy name. Possible values: {', '.join(AUTH_METHODS)}",
        example="default",
    )


class VendorCollectionRequest(BaseModel):
    vendor: str = Field(description="Vendor that hosts data", example="livechat")
    organization_id: str = Field(
        description=f"aka collection to use",
        example="f1ac8408-27b2-465e-89c6-b8708bfc262c",
    )


class VendorCollectionTokenRequest(VendorCollectionRequest):
    password: str = Field(description="This is for staff use")


class CollectionRequest(VendorCollectionRequest):
    query: str | None = Field(
        description="Query to generate answer for.", example="What is your name?", default=None
    )
    document_id: str | None = Field(
        description="Doc id. Use only if you know what this is.", default=None
    )
    doc_subcollection: str | None = Field(
        description="Subcollection where to look for document id. Required if document_id is presented.",
        default=None,
    )
    subcollections: List[str] | None = Field(
        description=f"Subcollections to use. Possible values: {', '.join(SubCollections['livechat'])}. Leave empty to use all subcollections.",
        example=["chatbot", "livechat"],
    )
    chat: List[dict] | None = Field(
        description="Optional: ongoing chat with the client",
        example=[
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "do you offer screen sharing chat"},
            {"role": "assistant", "content": "Hello, I will check, thanks for waiting."},
            {"role": "user", "content": "Sure."},
        ],
    )
    n_top_ranking: int | None = Field(
        description="Number of most relevant sources to be returned", default=3, example=3
    )

    # check that only query or document_id is provided
    @root_validator
    def check_query_or_document_id(cls, values):
        if not (bool(values.get("query")) ^ bool(values.get("document_id"))):
            raise ValueError("Provide either query or document_id")
        return values


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
    info_source: List[Dict] | None = Field(
        default=None,
        description="A list of dictionaries with information about the source of the answer.",
    )


class GetAnswerCollectionResponse(BaseModel):
    answer: str = Field(
        description="Answer to the query",
        example="I used to play drums.",
    )
    request_id: str = Field(
        description="A request id which is used to /set_reaction.",
        example="63cbd74e8d31a62a1512eab1",
    )
    source: List[Tuple[str, str, str]] | None = Field(
        default=None,
        description="A list of tuples (title, id, summary) with information about the source of the answer",
        example=[
            "Payment",
            "123456",
            "Payment methods and informaton summary. How to pay for subscription",
        ],
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


class HTTPExceptionResponse(BaseModel):
    detail: str = Field(example="Internal Server Error")
