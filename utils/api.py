import datetime
import logging
import traceback
from functools import wraps
from typing import List, Union

from fastapi import HTTPException, Request, status

from utils import CONFIG, DB


def log_get_answer(
    answer: str,
    context: str,
    document_ids: Union[str, List[str]],
    query: str,
    request: Request,
    api_version: str,
    vendor: str = None,
    organization_id: str = None,
    collections: List[str] = None,
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
        "organization_id": organization_id,
        "collections": collections,
    }
    request_id = DB[CONFIG["mongo"]["requests_collection"]].insert_one(row).inserted_id
    logging.info(row)
    return str(request_id)


def catch_errors(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if type(e) == HTTPException:
                raise e
            logging.error(f"{e.__class__.__name__}: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{e.__class__.__name__}: {e}",
            )

    return wrapper
