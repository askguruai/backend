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
from utils import CONFIG, DB, ml_requests, MILVUS_DB
from utils import ml_requests


class ChatsUploadHandler:
    def __init__(self, parser: ChatParser):
        self.parser = parser


    def handle_request(self, chats: List[dict], api_version: str, org_id: str, vendor: str) -> int:
        org_hash = hashlib.sha256(org_id.encode()).hexdigest()[:int(CONFIG["misc"]["hash_size"])]
        collection = MILVUS_DB.get_or_create_collection(f"{vendor}_{org_hash}_chats")

        all_chunks = []
        all_chunk_hashes = []
        all_doc_ids = []
        all_doc_titles = []

        for chat in chats:
            chunks, meta_info = self.parser.process_document(chat)
            new_chunks = []
            chunk_hashes = []
            for chunk in chunks:
                text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:int(CONFIG["misc"]["hash_size"])]
                res = collection.query(
                    expr=f"chunk_hash==\"{text_hash}\"",
                    offset=0,
                    limit=1,
                    output_fields=["chunk_hash"],
                    consistency_level="Strong"
                )
                if len(res) == 0:
                    # there is no such document yet, inserting
                    chunk_hashes.append(text_hash)
                    new_chunks.append(chunk)
            if len(new_chunks) == 0:
                # everyting is already in the database
                continue

            # all_embeddings.extend(embeddings)
            all_chunks.extend(new_chunks)
            all_doc_ids.extend([meta_info["chat_id"]] * len(new_chunks))
            all_doc_titles.extend([meta_info["chat_title"]] * len(new_chunks))
            all_chunk_hashes.extend(chunk_hashes)
        if len(all_chunks) != 0:
            all_embeddings = ml_requests.get_embeddings(all_chunks, api_version=api_version)
            data = [
                all_chunk_hashes,
                all_doc_ids,
                all_chunks,
                all_embeddings,
                all_doc_titles
            ]
            collection.insert(data)
            logging.info(f"Request of {len(chats)} chats inserted in database in {len(all_chunks)} chunks")
        
        return len(all_chunks)
