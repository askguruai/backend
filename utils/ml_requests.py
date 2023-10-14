import json
from typing import List, Union

import numpy as np
from aiohttp import FormData
from fastapi import UploadFile, status
from fastapi.encoders import jsonable_encoder
from loguru import logger
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from utils import CLIENT_SESSION_WRAPPER
from utils.errors import CoreMLError


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=120),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
async def get_transcript_from_file(file: UploadFile, api_version: str) -> str:
    data = FormData()
    data.add_field("file", await file.read(), filename=file.filename)
    async with CLIENT_SESSION_WRAPPER.coreml_session.post(
        f"/{api_version}/transcribe/",
        data=data,
    ) as response:
        response_status = response.status
        response_json = await response.json()
        if response_status == status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise CoreMLError(response_json["detail"])
        return response_json["data"]


@retry(
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=1, min=1, max=120),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
async def get_embeddings(chunks: List[str] | str, api_version: str) -> List[np.ndarray]:
    if type(chunks) is not list:
        chunks = [chunks]

    async with CLIENT_SESSION_WRAPPER.coreml_session.post(
        f"/{api_version}/embeddings/",
        json={"input": chunks},
    ) as response:
        # https://docs.aiohttp.org/en/stable/client_quickstart.html#response-content-and-status-code
        response_status = response.status
        response_json = await response.json()
        if response_status == status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise CoreMLError(response_json["detail"])
        embeddings = [np.array(emb) for emb in response_json["data"]]
        return embeddings


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
async def get_answer(
    context: str,
    query: str,
    api_version: str,
    mode: str = "general",
    chat: Union[list, None] = None,
    stream: bool = False,
    include_image_urls: bool = False,
    apply_formatting: bool = False,
) -> str:
    response = await CLIENT_SESSION_WRAPPER.coreml_session.post(
        f"/{api_version}/completions/",
        json={
            "info": context,
            "query": query,
            "mode": mode,
            "chat": jsonable_encoder(chat) if chat else None,
            "stream": stream,
            "include_image_urls": include_image_urls,
            "apply_formatting": apply_formatting,
        },
    )
    response_status = response.status
    if stream:
        answer = (
            json.loads(data.decode("utf-8"))["data"] if data else "" async for data, _ in response.content.iter_chunks()
        )
    else:
        response_json = await response.json()
        if response_status == status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise CoreMLError(response_json["detail"])
        answer = response_json["data"]
    return answer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
async def if_answer_in_context(
    context: str,
    query: str,
    api_version: str,
    chat: Union[list, None] = None,
) -> str:
    response = await CLIENT_SESSION_WRAPPER.coreml_session.post(
        f"/{api_version}/if_answer_in_context/",
        json={"info": context, "query": query, "chat": chat},
    )
    response_status = response.status
    response_json = await response.json()
    if response_status == status.HTTP_500_INTERNAL_SERVER_ERROR:
        raise CoreMLError(response_json["detail"])
    answer = response_json["answer"]
    return answer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
async def get_summary(
    info: str,
    max_tokens: int,
    api_version: str,
    stream: bool = False,
) -> str:
    response = await CLIENT_SESSION_WRAPPER.coreml_session.post(
        f"/{api_version}/summarization/",
        json={"info": info, "max_tokens": max_tokens, "stream": stream},
    )
    response_status = response.status
    if stream:
        answer = (
            json.loads(data.decode("utf-8"))["data"] if data else "" async for data, _ in response.content.iter_chunks()
        )
    else:
        response_json = await response.json()
        if response_status == status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise CoreMLError(response_json["detail"])
        answer = response_json["data"]
    return answer
