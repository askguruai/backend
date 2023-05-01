import hashlib
import logging
import pickle
from typing import List

from bson.binary import Binary
from bson.objectid import ObjectId

from handlers.collection_handler import CollectionHandler
from parsers import ChatParser
from utils import DB, ml_requests


class ChatsUploadHandler:
    def __init__(self, parser: ChatParser, collections_handler: CollectionHandler):
        self.parser = parser
        self.collection_handler = collections_handler

    async def handle_request(
        self, chats: List[dict], api_version: str, org_id: str, vendor: str
    ) -> int:
        for chat in chats:
            chunks, meta_info = self.parser.process_document(chat)
            embeddings = await ml_requests.get_embeddings(chunks, api_version=api_version)
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
                    logging.info(f"Chat {meta_info['chat_id']} chunk {i} inserted in the database")
        return len(chats)
