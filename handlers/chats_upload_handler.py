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

from handlers.collection_handler import CollectionHandler
from parsers import ChatParser
from utils import CONFIG, DB, ml_requests
from utils.ml_requests import get_embeddings
from utils.schemas import ResponseSourceChat


class ChatsUploadHandler:
    def __init__(self, parser: ChatParser, collections_handler: CollectionHandler):
        self.parser = parser
        self.collection_handler = collections_handler

    def handle_request(self, chats: List[dict], api_version: str, org_id: str, vendor: str) -> int:
        for chat in chats:
            chunks, meta_info = self.parser.process_document(chat)
            embeddings = get_embeddings(chunks, api_version=api_version)
            for i, pair in enumerate(zip(chunks, embeddings)):
                chunk, emb = pair
                text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:24]
                document = DB[f"{api_version}.collections.{vendor}.{org_id}.chats"].find_one(
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
                    DB[f"{api_version}.collections.{vendor}.{org_id}.chats"].insert_one(document)
                    self.collection_handler.update(
                        api_version=api_version,
                        vendor=vendor,
                        collection=org_id,
                        subcollection="chats",
                        data={"embedding": emb, "chunk": chunk,
                              "source": ResponseSourceChat(type="chat", chat_id=meta_info["chat_id"])},
                    )
                    logging.info(f"Chat {meta_info['chat_id']} chunk {i} inserted in the database")
        return len(chats)
