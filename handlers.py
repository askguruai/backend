import hashlib
import json
import logging
import os.path as osp
import shutil
import tempfile
from pathlib import Path
from pprint import pformat

import numpy as np
import requests
from utils.data import TextQueryRequest, PdfQueryRequest

from parsers import parse_text
from parsers.pdf_parser import extract_content, parse_pdf_content
from utils import CONFIG, STORAGE
from utils.ml import get_embeddings, get_answer, get_context_indices
from utils.errors import InvalidDocumentIdError
from typing import Tuple, Any, Union, Dict
from db import DB, GRIDFS
from fastapi import File, UploadFile


def text_query_handler(data: TextQueryRequest) -> Tuple[str, str, Any]:
    info = data.text if data.text else ""
    if info == "":
        context = ""
        doc_id = ""
        logging.info(f"Answering no-context question")
    else:
        processed_data = None
        if data.document_id is not None and data.document_id in STORAGE:
            # checking if doc_id presented and exists in storage
            processed_data = STORAGE[data.document_id]
            doc_id = data.document_id
            logging.info(f"Document ID presnted and document found: {data.document_id}")
        else:
            # otherwise, calculating hash and checking storage again
            info_hash = STORAGE.get_hash(info)
            logging.info(f"Hash of the input data: {info_hash}")
            if info_hash in STORAGE:
                processed_data = STORAGE[info_hash]
            doc_id = info_hash

        if processed_data is not None:
            logging.info(f"Cachefile found in storage by id {doc_id}")
            # if data loaded successfully from storage, just picking embeddings
            embeddings = []
            for chunk in processed_data:
                embeddings.append(chunk["embedding"])
            query_embedding = get_embeddings([data.query])[0]
        else:
            # data is new, need to parse and calculate embeddings
            # also need to save raw text to a designated collection in db
            row = {
                "document_id": doc_id,
                "text": data.text,
            }
            obj_id = DB[CONFIG["mongo"]["texts_collection"]].insert_one(row).inserted_id
            row = {
                "document_id": doc_id,
                "data_id": str(obj_id),
                "type": "txt",
            }
            DB[CONFIG["mongo"]["data_index_collection"]].insert_one(row)

            text_chunks = parse_text(info, int(CONFIG["text_handler"]["chunk_size"]))
            text_chunks.append(data.query)
            embeddings = get_embeddings(text_chunks)
            assert len(embeddings) == len(text_chunks)
            query = text_chunks.pop()
            query_embedding = embeddings.pop()
            processed_data = []
            for chunk, emb in zip(text_chunks, embeddings):
                processed_data.append({"text": chunk, "embedding": emb})
            STORAGE[doc_id] = processed_data

        logging.info(
            f"Number of chunks of size {CONFIG['text_handler']['chunk_size']}: {len(processed_data)}"
        )
        indices = get_context_indices(embeddings, query_embedding)
        context = "\n\n".join([processed_data[i]["text"] for i in indices])
        logging.info(f"Top {CONFIG['text_handler']['top_k_chunks']} chunks:\n{context}")
    answer = get_answer(context, data.query)
    return answer, context, doc_id


def pdf_upload_handler(file: UploadFile = File(...)) -> str:
    random_hash = STORAGE.get_hash()
    tempfile.TemporaryFile()
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = osp.join(tmpdir, f"{random_hash}.pdf")
        with open(fpath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        pdf_contents = extract_content(fpath)
        doc_id = STORAGE.get_hash(pdf_contents)
        if doc_id in STORAGE:
            # file with this exact content already exists and processed in storage
            logging.info(f"pdf_upload_handler: file found")
            pass
        else:
            text_chunks = parse_pdf_content(pdf_contents, int(CONFIG["text_handler"]["chunk_size"]))
            embeddings = get_embeddings(text_chunks)
            processed_data = []
            for chunk, emb in zip(text_chunks, embeddings):
                processed_data.append({"text": chunk, "embedding": emb})
            STORAGE[doc_id] = processed_data
            with open(fpath, "rb") as buf:
                obj_id = GRIDFS.put(buf)
            row = {
                "document_id": doc_id,
                "data_id": obj_id,
                "type": "pdf",
            }
            DB[CONFIG["mongo"]["data_index_collection"]].insert_one(row)
        return doc_id


def pdf_request_handler(data: PdfQueryRequest) -> Tuple[str, str]:
    doc_id = data.document_id
    if doc_id not in STORAGE:
        raise InvalidDocumentIdError()
    processed_data = STORAGE[doc_id]
    embeddings = []
    for chunk in processed_data:
        embeddings.append(chunk["embedding"])
    query_embedding = get_embeddings([data.query])[0]
    indices = get_context_indices(embeddings, query_embedding)
    context = "\n\n".join([processed_data[i]["text"] for i in indices])
    answer = get_answer(context, data.query)
    return answer, context
