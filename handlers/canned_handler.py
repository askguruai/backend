import time
from typing import List

from fastapi import HTTPException, status
from loguru import logger

from utils import AWS_TRANSLATE_CLIENT, MILVUS_DB, full_collection_name, ml_requests
from utils.misc import decode_security_code, int_list_encode
from utils.schemas import ApiVersion, CannedAnswer, MilvusSchema


class CannedHandler:
    async def add_canned_answer(
        self,
        api_version: ApiVersion,
        vendor: str,
        organization: str,
        collection: str,
        question: str,
        answer: str,
        security_groups: List[int] | None,
        timestamp: int | None,
        project_to_en: bool = True,
    ):
        collection_name = full_collection_name(vendor, organization, collection, is_canned=True)
        if MILVUS_DB.collection_status(full_collection_name(vendor, organization, collection)) == "NotExist":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Requested canned answer creation for collection '{collection}', but it does not exist in vendor '{vendor}' and organization '{organization}'!",
            )
        collection = MILVUS_DB.get_or_create_collection(collection_name, schema=MilvusSchema.CANNED_V0)
        # TODO: do translation!!!
        if project_to_en:
            question, answer = AWS_TRANSLATE_CLIENT.translate_text([question, answer])["translation"]
        question_embedding = (await ml_requests.get_embeddings(question, api_version.value))[0]
        security_code = int_list_encode(security_groups)
        timestamp = timestamp if timestamp is not None else int(time.time())
        mr = collection.insert([[question], [answer], [question_embedding], [timestamp], [security_code]])
        logger.info(f"Canned answer inserted with id {str(mr.primary_keys[0])}")
        return CannedAnswer(
            question=question,
            answer=answer,
            id=str(mr.primary_keys[0]),
            timestamp=timestamp,
            security_groups=security_groups,
        )

    async def get_canned_by_id(
        self, api_version: ApiVersion, vendor: str, organization: str, collection: str, canned_id: str
    ):
        collection_name = full_collection_name(vendor, organization, collection, is_canned=True)
        if (
            MILVUS_DB.collection_status(full_collection_name(vendor, organization, collection)) == "NotExist"
            or MILVUS_DB.collection_status(collection_name) == "NotExist"
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Either collection '{collection}' or its canned counterpart does not exist in vendor '{vendor}' and organization '{organization}'!",
            )
        collection = MILVUS_DB[collection_name]
        res = collection.query(
            expr=f'pk=={canned_id}',
            offset=0,
            limit=1,
            output_fields=["pk", "question", "answer", "timestamp", "security_groups"],
            consistency_level="Strong",
        )
        if len(res) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Canned answer with id {canned_id} was not found for collection {collection}!",
            )
        canned_result = CannedAnswer(
            id=res[0]["pk"],
            question=res[0]["question"],
            answer=res[0]["answer"],
            timestamp=res[0]["timestamp"],
            security_groups=decode_security_code(res[0]["security_groups"]),
        )
        return canned_result

    async def delete_canned_by_id(
        self, api_version: ApiVersion, canned_id: str, vendor: str, organization: str, collection: str
    ):
        # here are all the checks that collections exist and stuff
        existing_canned = await self.get_canned_by_id(api_version, vendor, organization, collection, canned_id)

        collection_name = full_collection_name(vendor, organization, collection, is_canned=True)
        collection = MILVUS_DB[collection_name]
        collection.delete(f"pk in [{existing_canned.id}]")

        return {"status": "ok"}

    async def update_canned_by_id(
        self,
        api_version: ApiVersion,
        canned_id: str,
        vendor: str,
        organization: str,
        collection: str,
        question: str | None,
        answer: str | None,
        security_groups: List[int] | None,
        timestamp: int | None,
        project_to_en: bool = True,
    ):
        existing_canned = await self.get_canned_by_id(api_version, vendor, organization, collection, canned_id)

        collection_name = full_collection_name(vendor, organization, collection, is_canned=True)
        m_collection = MILVUS_DB[collection_name]
        m_collection.delete(f"pk in [{existing_canned.id}]")

        question = question if question is not None else existing_canned.question
        answer = answer if answer is not None else existing_canned.answer
        timestamp = timestamp if timestamp is not None else existing_canned.timestamp
        security_groups = security_groups if security_groups is not None else existing_canned.security_groups

        return await self.add_canned_answer(
            api_version, vendor, organization, collection, question, answer, security_groups, timestamp, project_to_en
        )

    async def get_canned_collection(self, api_version: ApiVersion, vendor: str, organization: str, collection: str):
        collection_name = full_collection_name(vendor, organization, collection, is_canned=True)
        if (
            MILVUS_DB.collection_status(full_collection_name(vendor, organization, collection)) == "NotExist"
            or MILVUS_DB.collection_status(collection_name) == "NotExist"
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Either collection '{collection}' or its canned counterpart does not exist in vendor '{vendor}' and organization '{organization}'!",
            )
        collection = MILVUS_DB[collection_name]
        results = collection.query(
            expr=f'pk>0',
            offset=0,
            output_fields=["pk", "question", "answer", "timestamp", "security_groups"],
            consistency_level="Strong",
        )
        all_canned = []
        for res in results:
            all_canned.append(
                CannedAnswer(
                    id=res["pk"],
                    question=res["question"],
                    answer=res["answer"],
                    timestamp=res["timestamp"],
                    security_groups=decode_security_code(res["security_groups"]),
                )
            )
        return {"canned_answers": all_canned}
