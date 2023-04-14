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

from parsers import ChatParser
from utils import CONFIG, DB, ml_requests
from utils.ml_requests import get_embeddings


class ChatsUploadHandler:
    def __init__(self, parser: ChatParser):
        self.parser = parser

    def handle_request(self, chats: List[dict], api_version: str, collection: str) -> int:
        for chat in chats:
            chunks, meta_info = self.parser.process_document(chat)
            embeddings = get_embeddings(chunks, api_version=api_version)
            for i, pair in enumerate(zip(chunks, embeddings)):
                chunk, emb = pair
                text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:24]
                document = DB[f"{api_version}.collections.{collection}.chats"].find_one(
                    {"_id": ObjectId(text_hash)}
                )
                if not document:
                    document = {
                        "_id": ObjectId(text_hash),
                        # "link": f"https://help.groovehq.com/help/{meta_info['slug']}",
                        "doc_id": meta_info["chat_id"],
                        "chunk": chunk,
                        "embedding": Binary(pickle.dumps(emb)),
                    }
                    DB[f"{api_version}.collections.{collection}.chats"].insert_one(document)
                    logging.info(f"Chat {meta_info['chat_id']} chunk {i} inserted in the database")
        return len(chats)
