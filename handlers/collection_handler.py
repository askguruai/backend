import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import tiktoken
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger

from utils import AWS_TRANSLATE_CLIENT, CONFIG, MILVUS_DB, hash_string, ml_requests
from utils.errors import DatabaseError, DocumentAccessRestricted, InvalidDocumentIdError
from utils.misc import AsyncIterator, decode_security_code, int_list_encode
from utils.schemas import (
    ApiVersion,
    CannedAnswer,
    Collection,
    CollectionSolutionRequest,
    Document,
    GetCollectionAnswerResponse,
    GetCollectionRankingResponse,
    GetCollectionResponse,
    GetCollectionsResponse,
    Message,
    MilvusSchema,
    Role,
    Source,
)


class CollectionHandler:
    def __init__(self, top_k_chunks: int, chunk_size: int, tokenizer_name: str, max_tokens_in_context: int):
        self.top_k_chunks = top_k_chunks
        self.chunk_size = chunk_size
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.max_tokens_in_context = max_tokens_in_context

    async def get_answer(
        self,
        vendor: str,
        organization: str,
        collections: List[str],
        api_version: ApiVersion,
        user_security_groups: List[int],
        project_to_en: bool,
        query: str = None,
        document: str = None,
        document_collection: str = None,
        stream: bool = False,
        collections_only: bool = True,
        chat: List[Message] = None,
        include_image_urls: bool = False,
    ) -> GetCollectionAnswerResponse:
        orig_lang = "en"
        if query:
            if project_to_en:
                translation = AWS_TRANSLATE_CLIENT.translate_text(query)
                query = translation["translation"]
                orig_lang = translation["source_language"]
        else:
            # so it's chat
            if project_to_en:
                chat, orig_lang = AWS_TRANSLATE_CLIENT.translate_chat(chat=chat)
            query = "\n".join([msg.content for msg in chat if msg.role == Role.user])

        query_embedding = (await ml_requests.get_embeddings(query, api_version.value))[0]

        canned = MILVUS_DB.search_canned_collections(
            vendor=vendor, organization=organization, collections=collections, vec=query_embedding
        )

        if canned is not None:
            answer = canned["answer"]
            if orig_lang != "en" and project_to_en:
                answer = AWS_TRANSLATE_CLIENT.translate_text(answer, target_language=orig_lang, source_language="en")[
                    "translation"
                ]
            source = Source(
                id=canned["id"],
                title=canned["question"],
                relevance=canned["similarity"],
                collection=canned["collection"],
                is_canned=True,
            )
            if stream:
                answer = AsyncIterator([answer])
                response = (GetCollectionAnswerResponse(answer=text, sources=[source]) async for text in answer)
                return response, []
            else:
                return GetCollectionAnswerResponse(answer=answer, sources=[source]), []

        security_code = int_list_encode(user_security_groups)

        similarities, chunks, titles, doc_ids, doc_summaries, doc_collections = MILVUS_DB.search_collections_set(
            vendor,
            organization,
            collections,
            query_embedding,
            self.top_k_chunks,
            api_version,
            document_id_to_exclude=document,
            document_collection=document_collection,
            security_code=security_code,
        )
        mode = "support"
        if len(chunks) == 0:
            # todo: make a cache or sth
            answer = "Unable to find an answer"
            if orig_lang != "en":
                answer = AWS_TRANSLATE_CLIENT.translate_text(answer, target_language=orig_lang, source_language="en")[
                    "translation"
                ]
            return GetCollectionAnswerResponse(answer=answer, sources=[]), []

        context, i = "", 0
        while i < len(chunks) and len(self.enc.encode(context + chunks[i])) < self.max_tokens_in_context:
            if api_version == ApiVersion.v1:
                context += f"{chunks[i]}\n{'=' * 20}\n"
            elif api_version == ApiVersion.v2:
                context += f"---\ndoc_idx: {i}\n---\n{chunks[i]}\n{'=' * 20}\n"
                # context += f"---\ndoc_id: {i}\ndoc_collection: {doc_collections[i].split('_')[-1]}\n---\n{chunks[i]}\n{'=' * 20}\n"
            else:
                raise ValueError(f"Invalid api version: {api_version}")
            i += 1

        if not collections_only:
            answer_in_context = await ml_requests.if_answer_in_context(context, query, api_version)
            logger.info(f"answer_in_context: {answer_in_context}")
            if not answer_in_context:
                context, mode = "", "general"

        answer = await ml_requests.get_answer(
            context,
            query,
            api_version.value,
            mode=mode,
            stream=stream,
            chat=chat,
            include_image_urls=include_image_urls,
        )

        sources, seen = [], set()
        for title, doc_id, doc_summary, collection, relevance in zip(
            titles[:i], doc_ids[:i], doc_summaries[:i], doc_collections[:i], similarities[:i]
        ):
            if doc_id not in seen:
                sources.append(
                    Source(
                        id=doc_id,
                        title=title,
                        collection=collection.split("_")[-1],
                        summary=doc_summary,
                        relevance=relevance,
                        is_canned=False,
                    )
                )
                # we allow duplicate chunks on v2 because in context we index them as they appear
                if api_version == ApiVersion.v1 or vendor == "oneclickcx":
                    seen.add(doc_id)

        if stream:
            if orig_lang != "en" and project_to_en:
                if type(answer) != str:
                    answer_text = ""
                    async for text in answer:
                        answer_text += text
                else:
                    answer_text = answer

                answer = AWS_TRANSLATE_CLIENT.translate_text(
                    answer_text, target_language=orig_lang, source_language="en"
                )["translation"]

            if type(answer) == str:
                # this might happen either if we have a canned answer or after translation
                answer = AsyncIterator([answer])

            response = (GetCollectionAnswerResponse(answer=text, sources=sources) async for text in answer)
            return response, chunks[:i]

        if orig_lang != "en" and project_to_en:
            answer = AWS_TRANSLATE_CLIENT.translate_text(answer, target_language=orig_lang, source_language="en")[
                "translation"
            ]
        return GetCollectionAnswerResponse(answer=answer, sources=sources), chunks[:i]

    async def get_solution(
        self,
        vendor: str,
        organization: str,
        collections: List[str],
        document: str,
        document_collection: str,
        api_version: ApiVersion,
        user_security_groups: List[int],
        stream: bool = False,
        collection_only: bool = True,
    ) -> GetCollectionAnswerResponse:
        org_hash = hash_string(organization)
        security_code = int_list_encode(user_security_groups)
        embedding, query = self.get_data_from_id(
            document=document,
            full_collection_name=f"{vendor}_{org_hash}_{document_collection}",
            security_code=security_code,
        )

        # search_collections = [f"{vendor}_{org_hash}_{collection}" for collection in collections]

        _, chunks, titles, doc_ids, doc_summaries, doc_collections = MILVUS_DB.search_collections_set(
            vendor,
            organization,
            collections,
            embedding,
            self.top_k_chunks,
            api_version,
            document_id_to_exclude=document,
            document_collection=document_collection,
            security_code=security_code,
        )
        if len(chunks) == 0:
            return GetCollectionAnswerResponse(answer="Unable to find an answer :(", sources=[]), []

        context, i = "", 0
        while i < len(chunks) and len(self.enc.encode(context + chunks[i])) < self.max_tokens_in_context:
            if api_version == ApiVersion.v1:
                context += f"{chunks[i]}\n{'=' * 20}\n"
            elif api_version == ApiVersion.v2:
                context += f"---\ndoc_idx: {i}\n---\n{chunks[i]}\n{'=' * 20}\n"
                # context += f"---\ndoc_id: {i}\ndoc_collection: {doc_collections[i].split('_')[-1]}\n---\n{chunks[i]}\n{'=' * 20}\n"
            i += 1

        answer = await ml_requests.get_answer(context, query, api_version, stream=stream)

        sources, seen = [], set()
        for title, doc_id, doc_summary, collection in zip(
            titles[:i], doc_ids[:i], doc_summaries[:i], doc_collections[:i]
        ):
            if doc_id not in seen:
                sources.append(
                    Source(id=doc_id, title=title, collection=collection.split("_")[-1], summary=doc_summary)
                )
                # we allow duplicate chunks on v2 because in context we index them as they appear
                if api_version == ApiVersion.v1:
                    seen.add(doc_id)

        if stream:
            response = (GetCollectionAnswerResponse(answer=text, sources=sources) async for text in answer)
            return response, chunks[:i]

        return GetCollectionAnswerResponse(answer=answer, sources=sources), chunks[:i]

    def get_data_from_id(self, document: str, full_collection_name: str, security_code: int) -> np.ndarray:
        collection = MILVUS_DB[full_collection_name]
        res = collection.query(
            expr=f'doc_id=="{document}"',
            offset=0,
            limit=30,
            output_fields=["chunk", "emb_v1", "security_groups"],
            consistency_level="Strong",
        )
        if len(res) != 1:
            col_name = full_collection_name.rsplit("_", maxsplit=1)[-1]
            if len(res) == 0:
                text = f"Unable to retrieve document {document} in collection {col_name}"
            else:
                text = f"Ambiguous documents {document} in collection {col_name}"
            raise InvalidDocumentIdError(text)
        if res[0]["security_groups"] & security_code == 0:
            raise DocumentAccessRestricted(f"User has no access to document {document}")
        emb = res[0]["emb_v1"]
        query = res[0]["chunk"]
        query += "\n\nAdress the problem stated above"

        return emb, query

    def get_collections(self, vendor: str, organization: str, api_version: ApiVersion) -> GetCollectionResponse:
        organization_hash = hash_string(organization)
        collections = MILVUS_DB.get_collections(vendor, organization_hash)
        return GetCollectionsResponse(collections=[Collection(**collection) for collection in collections])

    def get_collection(
        self, vendor: str, organization: str, collection: str, api_version: ApiVersion, user_security_groups: List[int]
    ) -> GetCollectionResponse:
        organization_hash = hash_string(organization)
        security_code = int_list_encode(user_security_groups)
        full_collection_name = f"{vendor}_{organization_hash}_{collection}"
        try:
            milvus_collection = MILVUS_DB[full_collection_name]
        except DatabaseError:
            logger.error(
                f"Requested collection '{collection}' not found in vendor '{vendor}' and organization '{organization}'! Organization hash: {organization_hash}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Requested collection '{collection}' not found in vendor '{vendor}' and organization '{organization}'!",
            )
        chunks = milvus_collection.query(
            expr='pk >= 0',
            output_fields=["doc_id", "timestamp", "doc_title", "security_groups"],
        )

        documents = defaultdict(int)
        document_titles = {}
        for chunk in chunks:
            if chunk["security_groups"] & security_code:
                doc_id, timestamp, doc_title = chunk["doc_id"], chunk["timestamp"], chunk["doc_title"]
                documents[doc_id] = max(documents[doc_id], timestamp)
                document_titles[doc_id] = doc_title

        return GetCollectionResponse(
            documents=[
                Document(id=doc_id, title=document_titles[doc_id], timestamp=timestamp)
                for doc_id, timestamp in documents.items()
            ]
        )

    async def get_ranking(
        self,
        vendor: str,
        organization: str,
        top_k: int,
        api_version: ApiVersion,
        user_security_groups: List[int],
        query: str = None,
        document: str = None,
        document_collection: str = None,
        collections: List[str] = None,
        similarity_threshold=0,
        project_to_en=True,
    ) -> GetCollectionRankingResponse:
        organization_hash = hash_string(organization)
        security_code = int_list_encode(user_security_groups)
        if query:
            if project_to_en:
                translation = AWS_TRANSLATE_CLIENT.translate_text(text=query)
                query = translation["translation"]
            embedding = (await ml_requests.get_embeddings(query, api_version.value))[0]
        elif document:
            document_collection = document_collection or "faq"
            embedding, _ = self.get_data_from_id(
                document=document,
                full_collection_name=f"{vendor}_{organization_hash}_{document_collection}",
                security_code=security_code,
            )

        # extracting more than top_k chunks because each
        # document might be represented by several chunks
        # collections_search = [f"{vendor}_{organization_hash}_{collection}" for collection in collections]
        similarities, _, titles, doc_ids, doc_summaries, doc_collections = MILVUS_DB.search_collections_set(
            vendor,
            organization,
            collections,
            embedding,
            top_k * 5,
            api_version.value,
            document_id_to_exclude=document,
            document_collection=document_collection,
            security_code=security_code,
        )

        sources, seen, seen_title, i = [], set(), set(), 0
        while len(sources) < top_k and i < len(titles):
            if len(sources) == 0 or (
                similarities[i] > similarity_threshold and (titles[i] not in seen_title or doc_ids[i] not in seen)
            ):
                sources.append(
                    Source(
                        id=doc_ids[i],
                        title=titles[i],
                        collection=doc_collections[i].split("_")[-1],
                        summary=doc_summaries[i],
                        relevance=similarities[i],
                    )
                )
                seen.add(doc_ids[i])
                seen_title.add(titles[i])
            i += 1
        return GetCollectionRankingResponse(sources=sources)
