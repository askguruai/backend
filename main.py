import time
from typing import Any, Dict, List

import bson
import uvicorn
from aiohttp import ClientSession
from bson.objectid import ObjectId
from fastapi import Body, Depends, FastAPI, File, HTTPException, Path, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from loguru import logger
from pymongo.collection import ReturnDocument

from handlers import (
    CollectionHandler,
    DocumentHandler,
    DocumentsUploadHandler,
    LinkHandler,
    PDFUploadHandler,
    TextHandler,
)
from parsers import DocumentParser, DocumentsParser, LinkParser, TextParser
from utils import CLIENT_SESSION_WRAPPER, CONFIG, DB
from utils.api import catch_errors, log_get_answer, log_get_ranking, stream_and_log
from utils.auth import decode_token, get_livechat_token, get_organization_token, oauth2_scheme
from utils.filter_rules import archive_filter_rule, check_filters, create_filter_rule, get_filters, update_filter_rule
from utils.gunicorn_logging import run_gunicorn_loguru
from utils.schemas import (
    ApiVersion,
    Chat,
    CollectionDocumentsResponse,
    CollectionResponses,
    Doc,
    DocumentRequest,
    GetAnswerResponse,
    GetCollectionAnswerResponse,
    GetCollectionRankingResponse,
    GetCollectionResponse,
    GetCollectionsResponse,
    GetFiltersResponse,
    GetReactionsResponse,
    HTTPExceptionResponse,
    LikeStatus,
    LinkRequest,
    Log,
    NotFoundResponse,
    SetReactionRequest,
    TextRequest,
    UploadDocumentResponse,
)
from utils.uvicorn_logging import RequestLoggerMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLoggerMiddleware)


@app.on_event("startup")
async def init_handlers():
    global text_handler, link_handler, document_handler, pdf_upload_handler, collection_handler, documents_upload_handler, CLIENT_SESSION_WRAPPER
    CLIENT_SESSION_WRAPPER.coreml_session = ClientSession(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}"
    )
    CLIENT_SESSION_WRAPPER.general_session = ClientSession()
    text_handler = TextHandler(
        parser=TextParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
    )
    link_handler = LinkHandler(
        parser=LinkParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
    )
    document_handler = DocumentHandler(
        parser=DocumentParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
    )
    collection_handler = CollectionHandler(
        top_k_chunks=int(CONFIG["handlers"]["top_k_chunks"]),
        chunk_size=int(CONFIG["handlers"]["chunk_size"]),
        tokenizer_name=CONFIG["handlers"]["tokenizer_name"],
        max_tokens_in_context=int(CONFIG["handlers"]["max_tokens_in_context"]),
    )
    pdf_upload_handler = PDFUploadHandler(
        parser=DocumentParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
    )
    documents_upload_handler = DocumentsUploadHandler(
        parser=DocumentsParser(
            chunk_size=int(CONFIG["handlers"]["chunk_size"]), tokenizer_name=CONFIG["handlers"]["tokenizer_name"]
        ),
    )


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.get(
    "/{api_version}/info",
    response_model=Dict[str, Any],
    responses=CollectionResponses,
)
@catch_errors
async def get_info(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
):
    token_data = decode_token(token)
    return token_data


@app.post("/{api_version}/collections/token", responses=CollectionResponses)(get_organization_token)
@app.post("/{api_version}/collections/token_livechat", responses=CollectionResponses, include_in_schema=False)(
    get_livechat_token
)


######################################################
#                   COLLECTIONS                      #
######################################################


@app.get(
    "/{api_version}/collections",
    response_model=GetCollectionsResponse,
    responses=CollectionResponses,
)
@catch_errors
async def get_collections(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
):
    token_data = decode_token(token)
    response = collection_handler.get_collections(
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        api_version=api_version,
    )
    return response


@app.get(
    "/{api_version}/collections/answer",
    response_model=GetCollectionAnswerResponse,
    responses=CollectionResponses,
)
@catch_errors
async def get_collections_answer(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    collections: List[str] = Query(None, description="List of collections to search", example=["chats"]),
    query: str = Query(default=None, description="Query string", example="How to change my password?"),
    document: str = Query(default=None, description="Document ID", example="1234567890"),
    document_collection: str = Query(default=None, description="Document collection", example="chats"),
    stream: bool = Query(default=False, description="Stream results", example=False),
    collections_only: bool = Query(
        default=True,
        description="If True, the answer will be based only on collections in knowledge base. Otherwise, route will try to answer based on collections, but if it will not succeed it will try to generate answer from the model weights themselves.",
        example=True,
    ),
    user: str = Query(default=None, description="User ID", example="1234567890"),
):
    if not query and not document:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either query or document must be provided",
        )
    if bool(document) ^ bool(document_collection):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both document and document_collection must be provided",
        )
    token_data = decode_token(token)
    if not collections:
        collections = [
            collection.name
            for collection in collection_handler.get_collections(
                vendor=token_data["vendor"],
                organization=token_data["organization"],
                api_version=api_version,
            ).collections
        ]
    if document and not query:
        response = await collection_handler.get_solution(
            vendor=token_data["vendor"],
            organization=token_data["organization"],
            collections=collections,
            document=document,
            document_collection=document_collection,
            api_version=api_version,
            user_security_groups=token_data["security_groups"],
            stream=stream,
        )
    else:
        check_filters(vendor=token_data["vendor"], organization=token_data["organization"], query=query)
        response = await collection_handler.get_answer(
            vendor=token_data["vendor"],
            organization=token_data["organization"],
            collections=collections,
            query=query,
            api_version=api_version,
            user_security_groups=token_data["security_groups"],
            document=document,
            document_collection=document_collection,
            stream=stream,
            collections_only=collections_only,
        )
    request_id = log_get_answer(
        answer=response.answer if not stream else "",
        context="",
        document_ids=[source.id for source in response.sources] if not stream else [],
        query=query,
        request=request,
        api_version=api_version,
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        collections=collections,
        user=user,
    )
    if stream:
        return StreamingResponse(
            stream_and_log(response, request_id),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )
    else:
        response.request_id = request_id
        return response


@app.get(
    "/{api_version}/collections/ranking",
    response_model=GetCollectionRankingResponse,
    responses=CollectionResponses,
)
@catch_errors
async def get_collections_ranking(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    collections: List[str] = Query(None, description="List of collections to search"),
    query: str = Query(default=None, description="Query string", example="How to change my password?"),
    document: str = Query(default=None, description="Document ID", example="1234567890"),
    document_collection: str = Query(default=None, description="Document collection", example="chats"),
    top_k: int = Query(default=10, description="Number of top documents to return", example=10),
    similarity_threshold: float = Query(
        default=0.0, description="Similarity threshold to filter sources", example=0.75
    ),
    user: str = Query(default=None, description="User ID", example="1234567890"),
):
    # TODO add logging
    if bool(document) ^ bool(document_collection):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Both document and document_collection must be provided",
        )
    token_data = decode_token(token)
    # check_filters(vendor=token_data["vendor"], organization=token_data["organization"], query=query)
    if not collections:
        collections = [
            collection.name
            for collection in collection_handler.get_collections(
                vendor=token_data["vendor"],
                organization=token_data["organization"],
                api_version=api_version,
            ).collections
        ]
    response = await collection_handler.get_ranking(
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        collections=collections,
        top_k=top_k,
        api_version=api_version,
        query=query,
        document=document,
        document_collection=document_collection,
        user_security_groups=token_data["security_groups"],
        similarity_threshold=similarity_threshold,
    )
    request_id = log_get_ranking(
        document_ids=[source.id for source in response.sources],
        query=query,
        request=request,
        api_version=api_version,
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        collections=collections,
        user=user,
    )
    response.request_id = request_id
    return response


@app.get(
    "/{api_version}/collections/{collection}",
    response_model=GetCollectionResponse,
    responses=CollectionResponses | {status.HTTP_404_NOT_FOUND: {"model": NotFoundResponse}},
)
@catch_errors
async def get_collection(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    collection: str = Path(description="Collection within organization", example="chats"),
):
    token_data = decode_token(token)
    return collection_handler.get_collection(
        token_data["vendor"],
        token_data["organization"],
        collection,
        api_version,
        user_security_groups=token_data["security_groups"],
    )


@app.post(
    "/{api_version}/collections/{collection}",
    response_model=CollectionDocumentsResponse,
    responses=CollectionResponses,
)
@catch_errors
async def upload_collection_documents(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    collection: str = Path(description="Collection within organization", example="chats"),
    translate_to_en: bool = Body(
        True, description="Whether to translate uploaded documet into Eng (increases model performance)"
    ),
    summarize: bool = Body(
        False, description="Whether to summarize documents. Will override `summary` that is passed with the document"
    ),
    summary_length: int = Body(
        CONFIG["misc"]["default_summary_length"], description="Parameter controlling summarization lengt"
    ),
    documents: List[Doc] = Body(None, description="List of documents to upload"),
    chats: List[Chat] = Body(None, description="List of chats to upload"),
    links: List[str] = Body(None, description="Each link will be recursively crawled and uploaded"),
):
    # only one of documents, chats or links must be provided
    if sum([bool(documents), bool(chats), bool(links)]) != 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="One and only one of documents, chats or links must be provided",
        )
    token_data = decode_token(token)
    return await documents_upload_handler.handle_request(
        api_version=api_version,
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        collection=collection,
        documents=documents if documents else chats if chats else links,
        translate_to_en=translate_to_en,
        summarize=summarize,
        summary_length=summary_length,
    )


@app.delete(
    "/{api_version}/collections/{collection}",
    response_model=CollectionDocumentsResponse,
    responses=CollectionResponses | {status.HTTP_404_NOT_FOUND: {"model": NotFoundResponse}},
)
@catch_errors
async def upload_collection_documents(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    collection: str = Path(description="Collection within organization", example="chats"),
    documents: List[str] = Body(None, description="List of ids of documents to delete", example=["1234567890"]),
):
    token_data = decode_token(token)
    return await documents_upload_handler.delete_documents(
        api_version=api_version,
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        collection=collection,
        documents=documents,
    )


@app.get(
    "/{api_version}/reactions",
    responses=CollectionResponses,
    response_model=GetReactionsResponse,
)
async def get_reactions(request: Request, api_version: ApiVersion, token: str = Depends(oauth2_scheme)):
    token_data = decode_token(token)
    result = DB[CONFIG["mongo"]["requests_collection"]].find(
        {"vendor": token_data["vendor"], "organization": token_data["organization"]},
        {
            "datetime": 1,
            "query": 1,
            "answer": 1,
            "api_version": 1,
            "collections": 1,
            "rating": 1,
            "like_status": 1,
            "comment": 1,
            "user": 1,
        },
    )
    return GetReactionsResponse(
        reactions=[
            Log(
                id=str(row["_id"]),
                datetime=row["datetime"],
                user=row["user"] if "user" in row else None,
                query=row["query"],
                answer=row["answer"],
                api_version=row["api_version"],
                collections=row["collections"],
                rating=row["rating"] if "rating" in row else None,
                like_status=row["like_status"] if "like_status" in row else None,
                comment=row["comment"] if "comment" in row else None,
            )
            for row in result
        ]
    )


@app.post(
    "/{api_version}/reactions",
    responses=CollectionResponses,
)
async def upload_reaction(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    request_id: str = Body(..., description="Request ID to set reaction for.", example="63cbd74e8d31a62a1512eab1"),
    rating: int = Body(None, description="Rating to set. INT from 1 to 5.", example=5, gt=0, lt=6),
    like_status: LikeStatus = Body(None, description="Reaction to set.", example=LikeStatus.good_answer),
    answer_copied: bool = Body(None, description="Flag whether the answer was copied by user", example=True),
    comment: str = Body(None, description="Comment to set.", example="Very accurate!"),
):
    token_data = decode_token(token)
    if not (bool(rating) or bool(like_status) or bool(comment) or (answer_copied is not None)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either rating, like_status, comment or copy flag must be provided",
        )
    row_update = {}
    if rating is not None:
        row_update["rating"] = rating
    if like_status is not None:
        row_update["like_status"] = like_status
    if answer_copied is not None:
        row_update["answer_copied"] = answer_copied
    if comment is not None:
        row_update["comment"] = comment

    try:
        db_status = DB[CONFIG["mongo"]["requests_collection"]].find_one_and_update(
            {"_id": ObjectId(request_id), "vendor": token_data["vendor"], "organization": token_data["organization"]},
            {"$set": row_update},
            return_document=ReturnDocument.AFTER,
        )
        if not db_status:
            logger.error(f"Can't find row with id {request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Can't find row with id {request_id}",
            )
    except bson.errors.InvalidId as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return Response(status_code=status.HTTP_200_OK)


######################################################
#                    DEMO APP                        #
######################################################


@app.post(
    "/{api_version}/get_answer/text",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
    include_in_schema=False,
)
@catch_errors
async def get_answer_text(api_version: ApiVersion, text_request: TextRequest, request: Request):
    answer, context, document_id = await text_handler.get_answer(text_request, api_version.value)
    request_id = log_get_answer(answer, context, document_id, text_request.query, request, api_version.value)
    return GetAnswerResponse(answer=answer, request_id=request_id)


@app.post(
    "/{api_version}/get_answer/link",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
    include_in_schema=False,
)
@catch_errors
async def get_answer_link(api_version: ApiVersion, link_request: LinkRequest, request: Request):
    answer, context, document_id = await link_handler.get_answer(link_request, api_version.value)
    request_id = log_get_answer(answer, context, document_id, link_request.query, request, api_version.value)
    return GetAnswerResponse(answer=answer, request_id=request_id)


@app.post(
    "/{api_version}/get_answer/document",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
    include_in_schema=False,
)
@catch_errors
async def get_answer_document(api_version: ApiVersion, document_request: DocumentRequest, request: Request):
    answer, context, info_source, document_ids = await document_handler.get_answer(document_request, api_version.value)
    request_id = log_get_answer(answer, context, document_ids, document_request.query, request, api_version.value)
    return GetAnswerResponse(answer=answer, request_id=request_id, info_source=info_source)


@app.post(
    "/{api_version}/upload/pdf",
    response_model=UploadDocumentResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
    include_in_schema=False,
)
@catch_errors
async def upload_pdf(api_version: ApiVersion, file: UploadFile = File(...)):
    document_id = await pdf_upload_handler.process_file(file, api_version.value)
    return UploadDocumentResponse(document_id=document_id)


######################################################
#                    COMMON                          #
######################################################


@app.post(
    "/{api_version}/set_reaction",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": HTTPExceptionResponse},
        status.HTTP_404_NOT_FOUND: {"model": HTTPExceptionResponse},
    },
    include_in_schema=False,
)
async def set_reaction(api_version: ApiVersion, set_reaction_request: SetReactionRequest):
    row_update = {
        "like_status": set_reaction_request.like_status,
        "comment": set_reaction_request.comment,
    }

    try:
        db_status = DB[CONFIG["mongo"]["requests_collection"]].find_one_and_update(
            {"_id": ObjectId(set_reaction_request.request_id)},
            {"$set": row_update},
            return_document=ReturnDocument.AFTER,
        )
        if not db_status:
            logger.error(f"Can't find row with id {set_reaction_request.request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Can't find row with id {set_reaction_request.request_id}",
            )
    except bson.errors.InvalidId as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return Response(status_code=status.HTTP_200_OK)


######################################################
#                     FILTERS                        #
######################################################


@app.get("/{api_version}/filters", response_model=GetFiltersResponse)
@catch_errors
async def get_filter_rules_epoint(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
):
    token_data = decode_token(token)
    response = await get_filters(vendor=token_data["vendor"], organization=token_data["organization"])
    return response


@app.post(
    "/{api_version}/filters",
)
@catch_errors
async def create_filter_rule_epoint(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    name: str = Body(description="Rule name", example="Profanity"),
    description: str = Body(default=None, description="Rule name", example="No profanity allowed in requests"),
    stop_words: List[str] = Body(
        description="A list of words to be searched in requests", example=["damn", "sex", "paki"]
    ),
):
    token_data = decode_token(token)
    response = await create_filter_rule(
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        name=name,
        description=description,
        stop_words=stop_words,
    )
    return response


@app.patch(
    "/{api_version}/filters",
)
@catch_errors
async def update_filter_rule_epoint(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    name: str = Body(description="Rule name", example="Profanity"),
    description: str = Body(default=None, description="Rule name", example="No profanity allowed in requests"),
    stop_words: List[str] = Body(
        description="A list of words to be searched in requests", example=["damn", "sex", "paki"]
    ),
):
    token_data = decode_token(token)
    response = await update_filter_rule(
        vendor=token_data["vendor"],
        organization=token_data["organization"],
        name=name,
        description=description,
        stop_words=stop_words,
    )
    return response


@app.delete(
    "/{api_version}/filters",
)
@catch_errors
async def archive_filter_rule_epoint(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    name: str = Body(description="Rule name", example="Profanity"),
):
    token_data = decode_token(token)
    response = await archive_filter_rule(
        vendor=token_data["vendor"], organization=token_data["organization"], name=name
    )
    return response


if __name__ == '__main__':
    options = {
        "bind": CONFIG["app"]["host"] + ':' + CONFIG["app"]["port"],
        "workers": CONFIG["app"]["workers"],
        "timeout": CONFIG["app"]["timeout"],
    }
    run_gunicorn_loguru(app, options)
