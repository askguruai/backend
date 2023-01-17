import datetime
import logging
import shutil
from typing import List, Union

import bson
import uvicorn
from bson.objectid import ObjectId
from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pymongo.collection import ReturnDocument

from utils.errors import InvalidDocumentIdError, RequestDataModelMismatchError
from handlers import DocumentHandler, LinkHandler, TextHandler, PDFUploadHandler
from handlers.confluence_handler import search_request_handler
from parsers import DocumentParser, LinkParser, TextParser
from utils import CONFIG, DB
from utils.api import DocumentRequest, LinkRequest, SetReactionRequest, TextRequest, ConfluenceSearchRequest
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


@app.post("/get_answer/text")
async def get_answer_text(text_request: TextRequest, request: Request):
    answer, context, document_id = TEXT_HANDLER.get_answer(text_request)
    request_id = log_get_answer(answer, context, document_id, text_request.query, request)
    return {"data": answer, "request_id": request_id}


@app.post("/get_answer/link")
async def get_answer_link(link_request: LinkRequest, request: Request):
    answer, context, document_id = LINK_HANDLER.get_answer(link_request)
    request_id = log_get_answer(answer, context, document_id, link_request.query, request)
    return {"data": answer, "request_id": request_id}


@app.post("/get_answer/document")
async def get_answer_document(document_request: DocumentRequest, request: Request):
    try:
        answer, context, info_source, document_ids = DOCUMENT_HANDLER.get_answer(document_request)
    except (InvalidDocumentIdError, RequestDataModelMismatchError) as e:
        logging.error(f"Error happened: {e.__class__.__name__}")
        return e.response()
    request_id = log_get_answer(answer, context, document_ids, document_request.query, request)
    return {"data": answer, "request_id": request_id, "info_source": info_source}


@app.post("/get_answer/confluence")
async def get_answer_confluence(conf_request: ConfluenceSearchRequest, request: Request):
    search_request_handler(conf_request)


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    document_id = PDF_UPLOAD_HANDLER.process_file(file)
    return {"status": "success", "document_id": document_id}


@app.post("/set_reaction")
async def set_reaction(set_reaction_request: SetReactionRequest):
    row_update = {
        "like_status": set_reaction_request.like_status,
        "comment": set_reaction_request.comment,
    }

    try:
        status = DB[CONFIG["mongo"]["requests_collection"]].find_one_and_update(
            {"_id": ObjectId(set_reaction_request.request_id)},
            {"$set": row_update},
            return_document=ReturnDocument.AFTER,
        )
        result = (
            f"Row {set_reaction_request.request_id} was successfully updated"
            if status
            else f"Can't find row with id {set_reaction_request.request_id}"
        )
    except bson.errors.InvalidId as e:
        result = f"Provided id {set_reaction_request.request_id} has wrong format"

    logging.info(result)
    return {"result": result}


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
