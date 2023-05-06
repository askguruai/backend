import hashlib
import logging
from typing import Dict, List

from parsers import ChatParser
from utils import CONFIG, MILVUS_DB, hash_string, ml_requests
from utils.schemas import ApiVersion, UploadCollectionDocumentsResponse


class ChatsUploadHandler:
    def __init__(self, parser: ChatParser):
        self.parser = parser

    async def handle_request(
        self, api_version: str, vendor: str, organization: str, collection: str, chats: List[Dict]
    ) -> UploadCollectionDocumentsResponse:
        org_hash = hash_string(organization)
        collection = MILVUS_DB.get_or_create_collection(f"{vendor}_{org_hash}_{collection}")

        all_chunks = []
        all_chunk_hashes = []
        all_doc_ids = []
        all_doc_titles = []
        all_summaries = []
        all_timestamps = []
        for chat in chats:
            chunks, meta_info = self.parser.process_document(chat)
            chat_id = meta_info["doc_id"]
            existing_chunks = collection.query(
                expr=f'doc_id=="{chat_id}"',
                offset=0,
                limit=10000,
                output_fields=["pk", "chunk_hash"],
                consistency_level="Strong",
            )
            existing_chunks = {hit["chunk_hash"]: hit["pk"] for hit in existing_chunks}
            # determining which chunks are new
            new_chunks_hashes = []
            new_chunks = []
            for chunk in chunks:
                text_hash = hash_string(chunk)
                if text_hash in existing_chunks:
                    del existing_chunks[text_hash]
                else:
                    new_chunks.append(chunk)
                    new_chunks_hashes.append(text_hash)
            # dropping outdated chunks
            existing_chunks_pks = list(map(str, existing_chunks.values()))
            collection.delete(f"pk in [{','.join(existing_chunks_pks)}]")

            if len(new_chunks) == 0:
                # everyting is already in the database
                continue
            all_chunks.extend(new_chunks)
            all_doc_ids.extend([meta_info["doc_id"]] * len(new_chunks))
            all_doc_titles.extend([meta_info["doc_title"]] * len(new_chunks))
            all_summaries.extend([""] * len(new_chunks))
            all_chunk_hashes.extend(new_chunks_hashes)
            all_timestamps.extend([meta_info["timestamp"]] * len(new_chunks))
        if len(all_chunks) != 0:
            all_embeddings = await ml_requests.get_embeddings(all_chunks, api_version=api_version)
            data = [
                all_chunk_hashes,
                all_doc_ids,
                all_chunks,
                all_embeddings,
                all_doc_titles,
                all_summaries,
                all_timestamps,
            ]
            collection.insert(data)
            logging.info(f"Request of {len(chats)} chats inserted in database in {len(all_chunks)} chunks")

        return UploadCollectionDocumentsResponse(n_chunks=len(all_chunks))
