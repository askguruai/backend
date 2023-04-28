import logging
import shutil
from typing import List, Union

import bson
import requests
import uvicorn
from bson.objectid import ObjectId
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pymongo.collection import ReturnDocument

from handlers import (
    ChatsUploadHandler,
    CollectionHandler,
    DocumentHandler,
    LinkHandler,
    PDFUploadHandler,
    TextHandler,
)
from parsers import ChatParser, DocumentParser, LinkParser, TextParser
from utils import CONFIG, DB
from utils.api import catch_errors, log_get_answer
from utils.auth import get_org_collection_token, login, login_livechat, validate_auth_org_scope, validate_auth_default
from utils.errors import CoreMLError, InvalidDocumentIdError, RequestDataModelMismatchError
from utils.schemas import (
    ApiVersion,
    Collection,
    CollectionRequest,
    DocumentRequest,
    GetAnswerCollectionResponse,
    GetAnswerResponse,
    HTTPExceptionResponse,
    LinkRequest,
    SetReactionRequest,
    TextRequest,
    UploadChatsRequest,
    UploadChatsResponse,
    UploadDocumentResponse,
)
from utils.uvicorn_logging import run_uvicorn_loguru

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def init_handlers():
    global text_handler, link_handler, document_handler, pdf_upload_handler, collection_handler, chats_upload_handler
    text_handler = TextHandler(
        parser=TextParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
    )
    link_handler = LinkHandler(
        parser=LinkParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
    )
    document_handler = DocumentHandler(
        parser=DocumentParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
    )
    collection_handler = CollectionHandler(
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
        chunk_size=int(CONFIG["handlers"]["chunk_size"]),
    )
    pdf_upload_handler = PDFUploadHandler(
        parser=DocumentParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
    )
    chats_upload_handler = ChatsUploadHandler(
        parser=ChatParser(chunk_size=int(CONFIG["handlers"]["chunk_size"]))
    )


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


# fmt: off
@app.post("/token")(login)
@app.post("/token_livechat")(login_livechat)
@app.post("/godmode_token")(get_org_collection_token)
# fmt: on

@app.post(
    "/{api_version}/get_answer/collection",
    response_model=GetAnswerCollectionResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse},
        status.HTTP_401_UNAUTHORIZED: {"model": HTTPExceptionResponse},
    },
    dependencies=[Depends(validate_auth_org_scope)],
)
@catch_errors
async def get_answer_collection_deprecated(
    user_request: CollectionRequest,
    api_version: ApiVersion,
    request: Request,
):
    answer, context, source = collection_handler.get_answer(user_request, api_version.value)
    request_id = log_get_answer(
        answer=answer,
        context=context,
        document_ids=None,
        query=user_request.query,
        request=request,
        api_version=api_version.value,
        collection=user_request.organization_id,
        subcollections=user_request.subcollections,
    )
    return GetAnswerCollectionResponse(answer=answer, request_id=request_id, source=source)


@app.post(
    "/{api_version}/get_answer/text",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_answer_text(api_version: ApiVersion, text_request: TextRequest, request: Request):
    answer, context, document_id = text_handler.get_answer(text_request, api_version.value)
    request_id = log_get_answer(
        answer, context, document_id, text_request.query, request, api_version.value
    )
    return GetAnswerResponse(answer=answer, request_id=request_id)


@app.post(
    "/{api_version}/get_answer/link",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_answer_link(api_version: ApiVersion, link_request: LinkRequest, request: Request):
    answer, context, document_id = link_handler.get_answer(link_request, api_version.value)
    request_id = log_get_answer(
        answer, context, document_id, link_request.query, request, api_version.value
    )
    return GetAnswerResponse(answer=answer, request_id=request_id)


@app.post(
    "/{api_version}/get_answer/document",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def get_answer_document(
    api_version: ApiVersion, document_request: DocumentRequest, request: Request
):
    answer, context, info_source, document_ids = document_handler.get_answer(
        document_request, api_version.value
    )
    request_id = log_get_answer(
        answer,
        context,
        document_ids,
        document_request.query,
        request,
        api_version.value,
    )
    return GetAnswerResponse(answer=answer, request_id=request_id, info_source=info_source)


@app.post(
    "/{api_version}/upload/pdf",
    response_model=UploadDocumentResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
@catch_errors
async def upload_pdf(api_version: ApiVersion, file: UploadFile = File(...)):
    document_id = pdf_upload_handler.process_file(file, api_version.value)
    return UploadDocumentResponse(document_id=document_id)


@app.post(
    "/{api_version}/upload/chats/",
    response_model=UploadChatsResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse},
               status.HTTP_401_UNAUTHORIZED: {"model": HTTPExceptionResponse},
    },
    dependencies=[Depends(validate_auth_org_scope)],
)
@catch_errors
async def upload_chats(api_version: ApiVersion, user_request: UploadChatsRequest):
    processed_chats = chats_upload_handler.handle_request(chats=user_request.chats,
                                                          vendor=user_request.vendor,
                                                          org_id=user_request.organization_id,
                                                          api_version=api_version.value)
    return UploadChatsResponse(uploaded_chunks_number=str(processed_chats))


@app.post(
    "/{api_version}/set_reaction",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": HTTPExceptionResponse},
        status.HTTP_404_NOT_FOUND: {"model": HTTPExceptionResponse},
    },
)
async def set_reaction(api_version: ApiVersion, set_reaction_request: SetReactionRequest):
    row_update = {
        "like_status": set_reaction_request.like_status,
        "comment": set_reaction_request.comment,
    }

    try:
        db_status = DB[CONFIG["mongo"]["requests_collection"]].find_one_and_update(
            {"_id": ObjectId(set_reaction_request.request_id)},
            {"$set": row_update},
            return_document=ReturnDocument.AFTER,
        )
        if not db_status:
            logging.error(f"Can't find row with id {set_reaction_request.request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Can't find row with id {set_reaction_request.request_id}",
            )
    except bson.errors.InvalidId as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return Response(status_code=status.HTTP_200_OK)


if __name__ == "__main__":
    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
            ssl_certfile="/etc/certs/fullchain_askguru.pem",
            ssl_keyfile="/etc/certs/privkey_askguru.pem",
        )
    )
