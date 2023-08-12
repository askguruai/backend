import datetime
import traceback
from functools import wraps
from typing import List, Union

from bson.objectid import ObjectId
from fastapi import HTTPException, Request, status
from loguru import logger
from pymongo.collection import ReturnDocument

from utils import CONFIG, DB


async def stream_and_log(generator, request_id):
    answer, sources = "", []
    async for response in generator:
        answer += response.answer
        sources = response.sources
        response.request_id = request_id
        yield f"event: message\ndata: {response.json()}\n\n"
    logger.info(f"answer: {answer}")
    db_status = DB[CONFIG["mongo"]["requests_collection"]].find_one_and_update(
        {"_id": ObjectId(request_id)},
        {"$set": {"answer": answer, "document_id": [source.id for source in sources]}},
        return_document=ReturnDocument.AFTER,
    )


def log_get_ranking(
    document_ids: Union[str, List[str]],
    query: str,
    request: Request,
    api_version: str,
    vendor: str = None,
    organization: str = None,
    collections: List[str] = None,
    user: str = None,
) -> str:
    if isinstance(document_ids, str) == str:
        document_ids = [document_ids]
    row = {
        "ip": request.client.host,
        "datetime": datetime.datetime.utcnow(),
        "document_id": document_ids,
        "query": query,
        "api_version": api_version,
        "vendor": vendor,
        "organization": organization,
        "collections": collections,
        "user": user,
    }
    request_id = DB[CONFIG["mongo"]["requests_ranking_collection"]].insert_one(row).inserted_id
    logger.info(
        f"RANKING: {vendor}:{organization} over collections: {collections}, query: {query}, api_version: {api_version}, docs: {document_ids}"
    )
    return str(request_id)


def log_get_answer(
    answer: str,
    context: List[str],
    document_ids: Union[str, List[str]],
    query: str,
    request: Request,
    api_version: str,
    vendor: str = None,
    organization: str = None,
    collections: List[str] = None,
    user: str = None,
) -> str:
    if isinstance(document_ids, str) == str:
        document_ids = [document_ids]
    row = {
        "ip": request.client.host,
        "datetime": datetime.datetime.utcnow(),
        "document_id": document_ids,
        "query": query,
        "model_context": context,
        "answer": answer,
        "api_version": api_version,
        "vendor": vendor,
        "organization": organization,
        "collections": collections,
        "user": user,
    }
    request_id = DB[CONFIG["mongo"]["requests_collection"]].insert_one(row).inserted_id
    logger.info(
        f"vendor: {vendor}, organization: {organization}, collections: {collections}, query: {query}, api_version: {api_version}"
    )
    return str(request_id)


def catch_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if type(e) == HTTPException:
                raise e
            logger.error(f"{e.__class__.__name__}: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{e.__class__.__name__}: {e}",
            )

    return wrapper
