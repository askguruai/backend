import datetime
import logging

import bson
import uvicorn
from bson.objectid import ObjectId
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pymongo.collection import ReturnDocument

from db import DB
from handlers import text_query_handler, pdf_upload_handler as _pdf_upload_handler, pdf_request_handler
from utils import CONFIG
from utils.data import SetReactionRequest, TextQueryRequest, PdfQueryRequest
from utils.errors import *
from utils.logging import run_uvicorn_loguru

app = FastAPI()


origins = ["http://localhost:5858", "http://78.141.213.164/:5858", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get_answer")
async def get_answer(data: TextQueryRequest, request: Request):

    answer, context, doc_id = text_query_handler(data)

    row = {
        "ip": request.client.host,
        "datetime": datetime.datetime.utcnow(),
        "document_id": doc_id,
        "query": data.query,
        "model_context": context,
        "answer": answer,
    }
    request_id = DB[CONFIG["mongo"]["requests_collection"]].insert_one(row).inserted_id

    logging.info(f"Answer to the query: {answer}")
    return {"data": answer, "request_id": str(request_id), "document_id": str(doc_id)}


@app.post("/set_reaction")
async def set_reaction(set_reaction_request: SetReactionRequest):
    row_update = {
        "like_status": set_reaction_request.like_status,
        "comment": set_reaction_request.comment,
    }

    try:
        status = DB[CONFIG["mongo"]["collection"]].find_one_and_update(
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


@app.get("/upload_pdf")
async def upload_form():
    response = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>File Upload</title>
    </head>
    <body>
    <form action="/pdf_upload" method="post" enctype="multipart/form-data">
    
        <input type="file" name="file" id="file" accept="application/pdf">
        <input type="submit" value="Upload It" />
    </form>
    </body>
    </html>
    """
    return HTMLResponse(response)


@app.post("/pdf_upload")
async def pdf_upload_handler(file: UploadFile = File(...)):
    document_id = _pdf_upload_handler(file)
    return {
        "document_id": document_id
    }


@app.post("/pdf_answer")
async def pdf_answer(data: PdfQueryRequest, request: Request):
    try:
        answer, context = pdf_request_handler(data)
    except InvalidDocumentIdError as e:
        logging.error(f"Error happened: invalid document id {data.document_id}")
        return e.response()

    row = {
        "ip": request.client.host,
        "datetime": datetime.datetime.utcnow(),
        "document_id": data.document_id,
        "query": data.query,
        "model_context": context,
        "answer": answer,
    }
    request_id = DB[CONFIG["mongo"]["requests_collection"]].insert_one(row).inserted_id
    return {
        "data": answer, "request_id": str(request_id), "document_id": data.document_id
    }


if __name__ == "__main__":

    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
        )
    )
