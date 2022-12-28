import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse

from handlers import text_request_handler
from utils import CONFIG
from utils.data import TextRequest
from utils.logging import run_uvicorn_loguru

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/text_query")
async def text_encode(request: TextRequest):
    answer = text_request_handler(request)
    return {"data": answer}


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
