import hashlib
import logging
from typing import Dict, List

from fastapi import HTTPException, status
from loguru import logger
from tqdm import tqdm

from parsers import DocumentsParser
from utils import CONFIG, MILVUS_DB, hash_string, ml_requests
from utils.errors import DatabaseError
from utils.schemas import ApiVersion, Chat, CollectionDocumentsResponse, Doc


class DocumentsUploadHandler:
    def __init__(self, parser: DocumentsParser):
        self.parser = parser
        self.insert_chunk_size = 500

    async def handle_request(
        self,
        api_version: str,
        vendor: str,
        organization: str,
        collection: str,
        documents: List[Doc] | List[Chat] | List[str],
    ) -> CollectionDocumentsResponse:
        if isinstance(documents[0], str):
            # traversing each link, extracting all pages from each link,
            # representing them as docs and flatten the list
            documents = [doc for link in documents for doc in (await self.parser.link_to_docs(link))]

        org_hash = hash_string(organization)
        collection = MILVUS_DB.get_or_create_collection(f"{vendor}_{org_hash}_{collection}")

        all_chunks = []
        all_chunk_hashes = []
        all_doc_ids = []
        all_doc_titles = []
        all_summaries = []
        all_timestamps = []
        all_security_groups = []
        for doc in tqdm(documents):
            chunks, meta_info = self.parser.process_document(doc)
            doc_id = meta_info["doc_id"]
            existing_chunks = collection.query(
                expr=f'doc_id=="{doc_id}"',
                offset=0,
                limit=10000,
                output_fields=["pk", "chunk_hash", "security_groups", "timestamp"],
                consistency_level="Strong",
            )
            existing_chunks = {
                hit["chunk_hash"]: (hit["pk"], hit["security_groups"], hit["timestamp"]) for hit in existing_chunks
            }
            # determining which chunks are new
            new_chunks_hashes = []
            new_chunks = []
            for chunk in chunks:
                text_hash = hash_string(chunk)
                if (
                    text_hash in existing_chunks
                    and existing_chunks[text_hash][1] == meta_info["security_groups"]
                    # and existing_chunks[text_hash][2] >= meta_info["timestamp"]
                ):
                    del existing_chunks[text_hash]
                else:
                    new_chunks.append(chunk)
                    new_chunks_hashes.append(text_hash)
            # dropping outdated chunks
            existing_chunks_pks = list(map(lambda val: str(val[0]), existing_chunks.values()))
            collection.delete(f"pk in [{','.join(existing_chunks_pks)}]")

            if len(new_chunks) == 0:
                # everyting is already in the database
                continue
            all_chunks.extend(new_chunks)
            all_doc_ids.extend([meta_info["doc_id"]] * len(new_chunks))
            all_doc_titles.extend([meta_info["doc_title"]] * len(new_chunks))
            all_summaries.extend([meta_info["doc_summary"]] * len(new_chunks))
            all_chunk_hashes.extend(new_chunks_hashes)
            all_timestamps.extend([meta_info["timestamp"]] * len(new_chunks))
            all_security_groups.extend([meta_info["security_groups"]] * len(new_chunks))
        if len(all_chunks) != 0:
            all_embeddings = []
            for i in range(0, len(all_chunks), self.insert_chunk_size):
                all_embeddings.extend(
                    await ml_requests.get_embeddings(
                        all_chunks[i : i + self.insert_chunk_size], api_version=api_version
                    )
                )
                collection.insert(
                    [
                        all_chunk_hashes[i : i + self.insert_chunk_size],
                        all_doc_ids[i : i + self.insert_chunk_size],
                        all_chunks[i : i + self.insert_chunk_size],
                        all_embeddings[i : i + self.insert_chunk_size],
                        all_doc_titles[i : i + self.insert_chunk_size],
                        all_summaries[i : i + self.insert_chunk_size],
                        all_timestamps[i : i + self.insert_chunk_size],
                        all_security_groups[i : i + self.insert_chunk_size],
                    ]
                )
            logger.info(f"Request of {len(documents)} docs inserted in database in {len(all_chunks)} chunks")

        return CollectionDocumentsResponse(n_chunks=len(all_chunks))

    async def delete_documents(
        self,
        api_version: str,
        vendor: str,
        organization: str,
        collection: str,
        documents: List[str],
    ) -> CollectionDocumentsResponse:
        org_hash = hash_string(organization)
        full_collection_name = f"{vendor}_{org_hash}_{collection}"
        try:
            collection = MILVUS_DB[full_collection_name]
        except DatabaseError:
            logger.error(f"Requested collection {vendor}_{organization}_{collection} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Requested collection '{collection}' not found",
            )
        documents = [f"'{doc}'" for doc in documents]
        existing_chunks = collection.query(
            expr=f'doc_id in [{",".join(documents)}]',
            offset=0,
            limit=16384,
            output_fields=["pk"],
            consistency_level="Strong",
        )
        existing_chunks_pks = [str(hit["pk"]) for hit in existing_chunks]
        collection.delete(f"pk in [{','.join(existing_chunks_pks)}]")
        logger.info(f"Request of {len(documents)} docs deleted from database in {len(existing_chunks_pks)} chunks")
        return CollectionDocumentsResponse(n_chunks=len(existing_chunks_pks))
