import hashlib
import logging
import os.path as osp
import pickle
import random
import shutil
import tempfile
from typing import List

from bson.binary import Binary
from bson.objectid import ObjectId
from fastapi import File, UploadFile

from parsers import DocumentParser
from utils import CONFIG, DB, ml_requests, hash_string


class PDFUploadHandler:
    def __init__(self, parser: DocumentParser):
        self.parser = parser

    def get_embeddings_from_chunks(self, chunks: List[str], api_version: str) -> List[List[float]]:
        embeddings = ml_requests.get_embeddings(chunks, api_version)
        assert len(embeddings) == len(chunks)
        return embeddings

    def process_file(self, file: UploadFile, api_version: str) -> str:
        random_hash = str(hex(random.getrandbits(256)))[2:]
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = osp.join(tmpdir, f"{random_hash}.pdf")
            with open(fpath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            text = self.parser.get_text(fpath)
        text_hash = hash_string(text)
        # todo: switch to "exists" method or whatever
        document = DB[api_version + CONFIG["mongo"]["requests_inputs_collection"]].find_one(
            {"_id": ObjectId(text_hash)}
        )
        if document:
            logging.info(f"Document with hash {text_hash} found in database")
            return text_hash  # aka document_id

        sentences = self.parser.text_to_sentences(text)
        sentences = [sent.replace("\n", " ") for sent in sentences]
        chunks = self.parser.chunkise_sentences(sentences, int(CONFIG["handlers"]["chunk_size"]))
        embeddings = self.get_embeddings_from_chunks(chunks, api_version)
        document = {
            "_id": ObjectId(text_hash),
            "text": text,
            "chunks": chunks,
            "embeddings": Binary(pickle.dumps(embeddings)),
        }
        DB[api_version + CONFIG["mongo"]["requests_inputs_collection"]].insert_one(document)
        logging.info(f"Document with hash {text_hash} processed and inserted in the database")
        return text_hash  # aka document_id
