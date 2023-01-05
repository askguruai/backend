import datetime
import logging
import shutil

import bson
import uvicorn
from bson.objectid import ObjectId
from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pymongo.collection import ReturnDocument

from db import DB
from handlers import get_answer_from_info
from utils import CONFIG
from utils.data import QueryRequest, SetReactionRequest
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


@app.post("/text_query")
async def get_answer(query_request: QueryRequest, request: Request):

    print(query_request)
    info = query_request.text_input if query_request.text_input else ""
    # info += extract_text_from_link(query_request.link)
    # info += extract_text_from_doc(query_request.link)

    answer, context = get_answer_from_info(info, query_request.query)

    row = {
        "ip": request.client.host,
        "datetime": datetime.datetime.utcnow(),
        "text": query_request.text_input,
        "query": query_request.query,
        "model_context": context,
        "answer": answer,
    }
    request_id = DB[CONFIG["mongo"]["collection"]].insert_one(row).inserted_id

    logging.info(f"Answer to the query: {answer}")
    return {"data": answer, "request_id": str(request_id)}


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
    <form action="/file_upload" method="post" enctype="multipart/form-data">
    
        <input type="file" name="file" id="file" accept="application/pdf">
        <input type="submit" value="Upload It" />
    </form>
    </body>
    </html>
    """
    return HTMLResponse(response)


@app.post("/file_upload")
async def file_upload_handler(file: UploadFile = File(...)):
    with open("storage/file.pdf", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


if __name__ == "__main__":

    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
        )
    )
