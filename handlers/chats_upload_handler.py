import hashlib
import logging
from typing import List

from parsers import ChatParser
from utils import CONFIG, MILVUS_DB, ml_requests, hash_string


class ChatsUploadHandler:
    def __init__(self, parser: ChatParser):
        self.parser = parser

    async def handle_request(
        self, chats: List[dict], api_version: str, org_id: str, vendor: str
    ) -> int:
        org_hash = hash_string(org_id)
        collection = MILVUS_DB.get_or_create_collection(f"{vendor}_{org_hash}_chats")

        all_chunks = []
        all_chunk_hashes = []
        all_doc_ids = []
        all_doc_titles = []
        all_summaries = []
        for chat in chats:
            chunks, meta_info = self.parser.process_document(chat)
            chat_id = meta_info["doc_id"]
            existing_chunks = collection.query(
                expr=f'doc_id=="{chat_id}"',
                offset=0,
                limit=10000,
                output_fields=["chunk_hash"],
                consistency_level="Strong",
            )
            existing_chunks = set((hit["chunk_hash"] for hit in existing_chunks))
            # determining which chunks are new
            new_chunks_hashes = []
            new_chunks = []
            for chunk in chunks:
                text_hash = hash_string(chunk)
                if text_hash in existing_chunks:
                    existing_chunks.remove(text_hash)
                else:
                    new_chunks.append(chunk)
                    new_chunks_hashes.append(text_hash)
            # dropping outdated chunks
            existing_chunks = [f'"{ch}"' for ch in existing_chunks]
            collection.delete(f"chunk_hash in [{','.join(existing_chunks)}]")

            if len(new_chunks) == 0:
                # everyting is already in the database
                continue
            all_chunks.extend(new_chunks)
            all_doc_ids.extend([meta_info["doc_id"]] * len(new_chunks))
            all_doc_titles.extend([meta_info["doc_title"]] * len(new_chunks))
            all_summaries.extend([""] * len(new_chunks))
            all_chunk_hashes.extend(new_chunks_hashes)

        if len(all_chunks) != 0:
            all_embeddings = await ml_requests.get_embeddings(all_chunks, api_version=api_version)
            data = [
                all_chunk_hashes,
                all_doc_ids,
                all_chunks,
                all_embeddings,
                all_doc_titles,
                all_summaries,
            ]
            collection.insert(data)
            logging.info(
                f"Request of {len(chats)} chats inserted in database in {len(all_chunks)} chunks"
            )

        return len(all_chunks)
