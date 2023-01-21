import datetime
import logging
import shutil
from typing import List, Union

import bson
import requests
import uvicorn
from bson.objectid import ObjectId
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pymongo.collection import ReturnDocument

from handlers import DocumentHandler, LinkHandler, PDFUploadHandler, TextHandler
from parsers import DocumentParser, LinkParser, TextParser
from utils import CONFIG, DB
from utils.api import (
    DocumentRequest,
    GetAnswerResponse,
    HTTPExceptionResponse,
    LinkRequest,
    SetReactionRequest,
    TextRequest,
    UploadDocumentResponse,
)
from utils.errors import CoreMLError, InvalidDocumentIdError, RequestDataModelMismatchError
from utils.logging import run_uvicorn_loguru

app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEXT_HANDLER = TextHandler(
    parser=TextParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
    top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
)
LINK_HANDLER = LinkHandler(
    parser=LinkParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
    top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
)
DOCUMENT_HANDLER = DocumentHandler(
    parser=DocumentParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
    top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
)
PDF_UPLOAD_HANDLER = PDFUploadHandler(
    parser=DocumentParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def log_get_answer(
    answer: str, context: str, document_ids: Union[str, List[str]], query: str, request: Request
) -> str:
    if isinstance(document_ids, str) == str:
        document_ids = [document_ids]
    row = {
        "ip": request.client.host,
        "datetime": datetime.datetime.utcnow(),
        "document_id": document_ids,
        "query": query,
        "model_context": context,
        "answer": answer,
    }
    request_id = DB[CONFIG["mongo"]["requests_collection"]].insert_one(row).inserted_id
    logging.info(row)
    return str(request_id)


@app.post(
    "/get_answer/text",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
async def get_answer_text(text_request: TextRequest, request: Request):
    try:
        answer, context, document_id = TEXT_HANDLER.get_answer(text_request)
    except CoreMLError as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    request_id = log_get_answer(answer, context, document_id, text_request.query, request)
    return GetAnswerResponse(answer=answer, request_id=request_id)


@app.post(
    "/get_answer/link",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
async def get_answer_link(link_request: LinkRequest, request: Request):
    try:
        answer, context, document_id = LINK_HANDLER.get_answer(link_request)
    except (
        CoreMLError,
        requests.exceptions.MissingSchema,
        requests.exceptions.ConnectionError,
    ) as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    request_id = log_get_answer(answer, context, document_id, link_request.query, request)
    return GetAnswerResponse(answer=answer, request_id=request_id)


@app.post(
    "/get_answer/document",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
async def get_answer_document(document_request: DocumentRequest, request: Request):
    try:
        answer, context, info_source, document_ids = DOCUMENT_HANDLER.get_answer(document_request)
    except (InvalidDocumentIdError, RequestDataModelMismatchError, CoreMLError) as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    request_id = log_get_answer(answer, context, document_ids, document_request.query, request)
    return GetAnswerResponse(answer=answer, request_id=request_id, info_source=info_source)


@app.post(
    "/upload/pdf",
    response_model=UploadDocumentResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        document_id = PDF_UPLOAD_HANDLER.process_file(file)
    except CoreMLError as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    return UploadDocumentResponse(document_id=document_id)


@app.post(
    "/set_reaction",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": HTTPExceptionResponse},
        status.HTTP_404_NOT_FOUND: {"model": HTTPExceptionResponse},
    },
)
async def set_reaction(set_reaction_request: SetReactionRequest):
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
            ssl_certfile="/etc/certs/fullchain.pem",
            ssl_keyfile="/etc/certs/privkey.pem",
        )
    )
