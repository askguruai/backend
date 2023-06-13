from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple

from fastapi import status
from pydantic import BaseModel, Field

from utils.errors import TokenMalformedError


class ApiVersion(str, Enum):
    v1 = "v1"  # fast, w/o footnote links
    v2 = "v2"  # slow, w/ footnote links


class HTTPExceptionResponse(BaseModel):
    detail: str = Field(example="Internal Server Error")


class AuthExceptionResponse(BaseModel):
    detail: str = Field(example="Could not validate credentials")

class NotFoundResponse(BaseModel):
    detail: str = Field(example="Requested resource not found")


CollectionResponses = {
    status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse},
    status.HTTP_401_UNAUTHORIZED: {"model": AuthExceptionResponse},
}

AUTH_METHODS = ["org_scope", "default"]


class Document(BaseModel):
    id: str = Field(description="Id of the document", example="7af8c3e548e40aeb984c42dd")
    timestamp: int = Field(description="Document last change time", example=1623345600)


class Collection(BaseModel):
    name: str = Field(description="Name of the collection", example="tickets")
    n_documents: int = Field(description="Number of documents in the collection", example=100)


class User(BaseModel):
    id: str = Field(description="Id of the user", example="7af8c3e548e40aeb984c42dd")
    name: str = Field(description="Name of the user", example="John Doe")


class Message(BaseModel):
    role: str = Field(description="Role of the user", example="assistant")
    content: str = Field(description="Content of the message", example="Hello, how can I help you?")


class Chat(BaseModel):
    id: str = Field(description="Id of the chat", example="7af8c3e548e40aeb984c42dd")
    timestamp: int = Field(description="Chat last change time", example=1623345600)
    user: User = Field(description="User of the chat", example=User(id="7af8c3e548e40aeb984c42dd", name="John Doe"))
    history: List[Message] = Field(
        description="History of the chat", example=[Message(role="assistant", content="Hello, how can I help you?")]
    )
    security_groups: List[int] = Field(description="Security groups of the chat", example=[0, 2])


class Doc(BaseModel):
    content: str = Field(description="Content of the document", example="")
    id: str | None = Field(description="Id of the document", example="7af8c3e548e40aeb984c42dd")
    title: str | None = Field(description="Title of the document", example="Passwords")
    summary: str | None = Field(description="Summary of the document", example="Instruction when forget password")
    timestamp: int | None = Field(description="Document last change time", example=1623345600)
    security_groups: List[int] | None = Field(description="Security groups of the document", example=[0, 2])


class GetCollectionsResponse(BaseModel):
    collections: List[Collection] = Field(
        description="Dict of collections names and number of documents in them",
        example=[
            Collection(
                name="tickets",
                n_documents=100,
            ),
        ],
    )


class GetCollectionResponse(BaseModel):
    documents: List[Document] = Field(
        description="List of documents from given collection",
        example=[
            Document(
                id="7af8c3e548e40aeb984c42dd",
                timestamp=1623345600,
            )
        ],
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


class GetCollectionAnswerRequest(VendorCollectionRequest):
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
            {"role": "assistant", "content": "Hello, I will check, thanks for waiting."},
            {"role": "user", "content": "Sure."},
        ],
    )


class CollectionSolutionRequest(VendorCollectionRequest):
    document_id: str = Field(description="Doc id. Use only if you know what this is.", default=None)
    document_collection: str = Field(
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


class GetCollectionAnswerResponse(BaseModel):
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
    document_id: str = Field(description="ID of an uploaded document.", example="7af8c3e548e40aeb984c42dd")


class UploadCollectionDocumentsResponse(BaseModel):
    n_chunks: str = Field(description="Number of chunks successfully uploaded", example="5")


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
    comment: str | None = Field(default=None, description="Comment to set.", example="Very accurate!")

    class Config:
        use_enum_values = True


class Log(BaseModel):
    id: str
    datetime: datetime
    query: str
    answer: str
    api_version: str
    collections: List[str]
    user: str | None
    rating: int | None
    like_status: LikeStatus | None
    comment: str | None


class GetReactionsResponse(BaseModel):
    reactions: List[Log] = Field(
        description="List of reactions",
        example=[
            Log(
                id="123456",
                datetime=datetime(2021, 3, 1, 12, 0, 0),
                query="What is your name?",
                answer="My name is Bob",
                api_version=ApiVersion.v1,
                collections=["chats", "tickets"],
                rating=5,
                like_status=LikeStatus.good_answer,
                comment="Very accurate!",
            )
        ],
    )


class PostFilterResponse(BaseModel):
    name: str = Field(description="Rule name that was sccessfully added/updated/deleted", example="ProfanityRule")


class FilterRule(BaseModel):
    name: str = Field(description="Unique rule name", example="ProfanityRule")
    description: str | None = Field(description="Optionanl rule description", example="Profanity is prohibited")
    stop_words: List[str] = Field(description="Rule stop words", example=["damn", "sex"])
    timestamp: int = Field(description="Timestamp of last operation (creation, update, archive)")


class GetFiltersResponse(BaseModel):
    active_rules: List[FilterRule] = Field(
        description="Organization active rules list",
        example=[
            {
                "name": "ProfanityRule",
                "description": "Optionanl rule description",
                "stop_words": ["damn", "sex"],
                "timestamp": 12345,
            },
            {
                "name": "NoRacism",
                "description": "Racist language is forbidden",
                "stop_words": ["paki", "nword"],
                "timestamp": 456789,
            },
        ],
    )
    archived_rules: List[FilterRule] = Field(
        description="Organization archived rules list",
        example=[
            {
                "name": "HateCitrus",
                "description": "Enough of them",
                "stop_words": ["orange", "lemon", "lime"],
                "timestamp": 456789,
            }
        ],
    )
