import logging
from typing import Any, Dict, List

import bson
import uvicorn
from aiohttp import ClientSession
from bson.objectid import ObjectId
from fastapi import Body, Depends, FastAPI, File, HTTPException, Path, Query, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pymongo.collection import ReturnDocument

from handlers import ChatsUploadHandler, CollectionHandler, DocumentHandler, LinkHandler, PDFUploadHandler, TextHandler
from parsers import ChatParser, DocumentParser, LinkParser, TextParser
from utils import CLIENT_SESSION_WRAPPER, CONFIG, DB
from utils.api import catch_errors, log_get_answer
from utils.auth import (
    decode_token,
    get_livechat_token,
    get_organization_token,
    oauth2_scheme,
    validate_organization_scope,
)
from utils.schemas import (
    ApiVersion,
    CollectionResponses,
    DocumentRequest,
    GetAnswerResponse,
    GetCollectionAnswerResponse,
    GetCollectionRankingResponse,
    GetCollectionResponse,
    HTTPExceptionResponse,
    LinkRequest,
    SetReactionRequest,
    TextRequest,
    UploadCollectionDocumentsResponse,
    UploadDocumentResponse,
)
from utils.uvicorn_logging import RequestLoggerMiddleware, run_uvicorn_loguru

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
    global text_handler, link_handler, document_handler, pdf_upload_handler, collection_handler, chats_upload_handler, CLIENT_SESSION_WRAPPER
    CLIENT_SESSION_WRAPPER.coreml_session = ClientSession(f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}")
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
    )
    pdf_upload_handler = PDFUploadHandler(
        parser=DocumentParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
    )
    chats_upload_handler = ChatsUploadHandler(
        parser=ChatParser(chunk_size=int(CONFIG["handlers"]["chunk_size"])),
    )


@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")


@app.post("/{api_version}/livechat/token", responses=CollectionResponses)(get_livechat_token)
@app.post("/{api_version}/{vendor}/{organization}/token", responses=CollectionResponses)(get_organization_token)


######################################################
#                   COLLECTIONS                      #
######################################################


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


@app.get(
    "/{api_version}/{vendor}/{organization}/answer",
    response_model=GetCollectionAnswerResponse,
    responses=CollectionResponses,
    dependencies=[Depends(validate_organization_scope)],
)
@catch_errors
async def get_collection_answer(
    request: Request,
    api_version: ApiVersion,
    vendor: str,
    organization: str,
    token: str = Depends(oauth2_scheme),
    # TODO make not mandatory collections
    collections: List[str] = Query(description="List of collections to search", example=["chats"]),
    query: str = Query(default=None, description="Query string", example="How to change my password?"),
    document: str = Query(default=None, description="Document ID", example="1234567890"),
    document_collection: str = Query(default=None, description="Document collection", example="chats"),
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
    if query and not document:
        response = await collection_handler.get_answer(
            vendor=vendor,
            organization=organization,
            collections=collections,
            query=query,
            api_version=api_version,
            user_security_groups=token_data["security_groups"],
        )
    elif document and not query:
        response = await collection_handler.get_solution(
            vendor=vendor,
            organization=organization,
            collections=collections,
            document=document,
            document_collection=document_collection,
            api_version=api_version,
            user_security_groups=token_data["security_groups"],
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only one of query or document must be provided",
        )
    request_id = log_get_answer(
        response.answer,
        "",
        [source.id for source in response.sources],
        query,
        request,
        api_version,
        vendor,
        organization,
        collections,
    )
    response.request_id = request_id
    return response


@app.get(
    "/{api_version}/{vendor}/{organization}/ranking",
    response_model=GetCollectionRankingResponse,
    responses=CollectionResponses,
    dependencies=[Depends(validate_organization_scope)],
)
@catch_errors
async def get_collection_ranking_query(
    request: Request,
    api_version: ApiVersion,
    vendor: str,
    organization: str,
    token: str = Depends(oauth2_scheme),
    # TODO make not mandatory collections
    collections: List[str] = Query(description="List of collections to search"),
    query: str = Query(default=None, description="Query string", example="How to change my password?"),
    document: str = Query(default=None, description="Document ID", example="1234567890"),
    document_collection: str = Query(default=None, description="Document collection", example="chats"),
    top_k: int = Query(default=10, description="Number of top documents to return", example=10),
):
    # TODO add logging
    if not (bool(query) ^ bool(document)):
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
    return await collection_handler.get_ranking(
        vendor=vendor,
        organization=organization,
        collections=collections,
        top_k=top_k,
        api_version=api_version,
        query=query,
        document=document,
        document_collection=document_collection,
        user_security_groups=token_data["security_groups"],
    )


@app.get(
    "/{api_version}/{vendor}/{organization}/{collection}",
    response_model=GetCollectionResponse,
    responses=CollectionResponses,
    dependencies=[Depends(validate_organization_scope)],
)
@catch_errors
async def get_collection(
    request: Request,
    api_version: ApiVersion,
    token: str = Depends(oauth2_scheme),
    vendor: str = Path(description="Vendor name", example="livechat"),
    organization: str = Path(description="Organization within vendor", example="f1ac8408-27b2-465e-89c6-b8708bfc262c"),
    collection: str = Path(description="Collection within organization", example="chats"),
):
    token_data = decode_token(token)
    return collection_handler.get_collection(
        vendor, organization, collection, api_version, user_security_groups=token_data["security_groups"]
    )


@app.post(
    "/{api_version}/{vendor}/{organization}/{collection}",
    response_model=UploadCollectionDocumentsResponse,
    responses=CollectionResponses,
    dependencies=[Depends(validate_organization_scope)],
)
@catch_errors
async def upload_collection_documents(
    request: Request,
    api_version: ApiVersion,
    vendor: str = Path(description="Vendor name", example="livechat"),
    organization: str = Path(description="Organization within vendor", example="f1ac8408-27b2-465e-89c6-b8708bfc262c"),
    collection: str = Path(description="Collection within organization", example="chats"),
    documents: List[Dict] = Body(None, description="List of documents to upload"),
    chats: List[Dict] = Body(None, description="List of chats to upload"),
):
    if documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Documents are not supported yet",
        )
    return await chats_upload_handler.handle_request(
        api_version=api_version,
        vendor=vendor,
        organization=organization,
        collection=collection,
        chats=chats,
    )


######################################################
#                    DEMO APP                        #
######################################################


@app.post(
    "/{api_version}/get_answer/text",
    response_model=GetAnswerResponse,
    responses={status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": HTTPExceptionResponse}},
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
            logging.error(f"Can't find row with id {set_reaction_request.request_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Can't find row with id {set_reaction_request.request_id}",
            )
    except bson.errors.InvalidId as e:
        logging.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return Response(status_code=status.HTTP_200_OK)


if __name__ == "__main__":
    run_uvicorn_loguru(
        uvicorn.Config(
            "main:app",
            host=CONFIG["app"]["host"],
            port=int(CONFIG["app"]["port"]),
            log_level=CONFIG["app"]["log_level"],
        )
    )
