import datetime
import logging
import shutil

import uvicorn
from bson.objectid import ObjectId
from fastapi import FastAPI, File, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse

from db import DB
from handlers import text_request_handler
from utils import CONFIG
from utils.data import SetReactionRequest, TextRequest
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
async def text_query(text_request: TextRequest, request: Request):

    answer, context = text_request_handler(text_request)

    row = {
        "ip": request.client.host,
        "datetime": datetime.datetime.utcnow(),
        "text": text_request.text_input,
        "query": text_request.query,
        "model_context": context,
        "answer": answer,
    }
    request_id = DB[CONFIG["mongo"]["collection"]].insert_one(row).inserted_id

    logging.info(f"Answer to the query: {answer}")
    return {"data": answer, "request_id": str(request_id)}


@app.post("/set_reaction")
async def set_reaction(set_reaction_request: SetReactionRequest):
    row_update = {"like": set_reaction_request.like, "comment": set_reaction_request.comment}

    DB[CONFIG["mongo"]["collection"]].find_one_and_update(
        {"_id": ObjectId(set_reaction_request.request_id)}, {"$set": row_update}
    )


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
