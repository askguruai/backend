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
from milvus_db.utils import get_or_create_collection


class ChatsUploadHandler:
    def __init__(self, parser: ChatParser):
        self.parser = parser


    def handle_request(self, chats: List[dict], api_version: str, org_id: str, vendor: str) -> int:
        org_hash = hashlib.sha256(org_id.encode()).hexdigest()[:24]
        collection = get_or_create_collection(f"{vendor}_{org_hash}_chats")

        all_chunks = []
        all_hash_ids = []
        all_doc_ids = []
        all_doc_titles = []

        for chat in chats:
            chunks, meta_info = self.parser.process_document(chat)
            new_chunks = []
            id_hashes = []
            for chunk in chunks:
                text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:24]
                res = collection.query(
                    expr=f"hash_id==\"{text_hash}\"",
                    offset=0,
                    limit=1,
                    output_fields=["hash_id"],
                    consistency_level="Strong"
                )
                if len(res) == 0:
                    # there is no such document yet, inserting
                    id_hashes.append(text_hash)
                    new_chunks.append(chunk)
            if len(new_chunks) == 0:
                # everyting is already in the database
                continue

            # embeddings = get_embeddings(new_chunks, api_version=api_version)

            # all_embeddings.extend(embeddings)
            all_chunks.extend(new_chunks)
            all_doc_ids.extend([meta_info["chat_id"]] * len(new_chunks))
            all_doc_titles.extend([meta_info["chat_title"]] * len(new_chunks))
            all_hash_ids.extend(id_hashes)
        if len(all_chunks) != 0:
            all_embeddings = get_embeddings(all_chunks, api_version=api_version)
            data = [
                all_hash_ids,
                all_doc_ids,
                all_chunks,
                all_embeddings,
                all_doc_titles
            ]
            collection.insert(data)
            logging.info(f"Request of {len(chats)} chats inserted in database in {len(all_chunks)} chunks")
        
        return len(chats)
