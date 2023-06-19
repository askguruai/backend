import os
import os.path as osp
import sys

sys.path.insert(1, os.getcwd())

import asyncio
import glob
import json
from argparse import ArgumentParser
from typing import List

from aiohttp import ClientSession
from loguru import logger
from pymilvus import utility

from parsers import DocxParser
from utils import CLIENT_SESSION_WRAPPER, CONFIG, MILVUS_DB, hash_string, ml_requests
from utils.tokenize_ import doc_to_chunks

DOCS_N_LINKS = {
    "How to Add New Contact Number.docx": "https://docs.google.com/document/d/1srUhSp5CbafXIslVKnJFiZ6tUinsIgG7/",
    "How to Add or login to Google Account.docx": "https://docs.google.com/document/d/1AxkZfGlRpQ3wQhwhWTV_BgFlO5bJllR3/",
    "How to Add Widget.docx": "https://docs.google.com/document/d/1UvkHEiVmICfaFoH3fEM2r3sm176lTuTI/",
    "How to Adjust Sound and Notification Volume.docx": "https://docs.google.com/document/d/1YYgVbQjR0Vo2Q5srAW1hbchG6uXFEhyn/",
    "How to allow apps to show bubbles.docx": "https://docs.google.com/document/d/15mZk6bTPL6cfPYU-YWtI0WD1eCNgRsla/",
    "How to Browser Files.docx": "https://docs.google.com/document/d/1NkpF1tGa-wIB7CQcP5CzFFAnyGbXFMB8/",
    "How to Change the Password or PIN or Lock Type.docx": "https://docs.google.com/document/d/1pEBKpYt3qijtW_WuEkGMyJv0kAyivqaD/",
    "How to connect external display.docx": "https://docs.google.com/document/d/1VMZYK3LV6PdPukaJxVTPe-6wx2ISrDFf/",
    "How to connect to Wi-Fi.docx": "https://docs.google.com/document/d/1TIZ_bNMLW3_UcHxcayjv8KmY5OCqgyw3/",
    "How to Enable Dark Theme.docx": "https://docs.google.com/document/d/1QMYIKTSUxiOT5g7_GxJc5PEkLyTq6O7X/",
    "How to Enable Developer Option.docx": "https://docs.google.com/document/d/1QbG8NapLdxpP43FN5mYvdb2f4_R0Rf-B/",
    "How to Factory Reset.docx": "https://docs.google.com/document/d/1lm_lETldfiWBUtB394qPCjhfBpN1KxNJ/",
    "How to insert SIM Card.docx": "https://docs.google.com/document/d/1-8ZHNJmgfXilh92wf4inY5sw8Ii5neot/",
    "How to move or copy files and folders.docx": "https://docs.google.com/document/d/19JGnCRWPDTcryuCIwaqd5SbKEwaygyNr/",
    "How to use touchpad.docx": "https://docs.google.com/document/d/1qQPO0ttjb8w8dOkCxAYYy1Kn-6Y6jbqS/",
}


async def process_file(filepath: str, parser: DocxParser, collection, api_version):
    filename = osp.split(filepath)[1]
    doc_link = DOCS_N_LINKS[filename]
    chunks, meta_info = parser.process_file(filepath)
    if len(chunks) == 0:
        return True  # nothing to do
    doc_id = DOCS_N_LINKS[filename]
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
        return True
    embeddings = await ml_requests.get_embeddings(new_chunks, api_version=api_version)

    data = [
        new_chunks_hashes,
        [doc_link] * len(new_chunks),
        new_chunks,
        embeddings,
        [meta_info["doc_title"]] * len(new_chunks),
        [""] * len(new_chunks),
        [0] * len(new_chunks),
        [2**63 - 1] * len(new_chunks),
    ]
    collection.insert(data)
    logger.info(f"Document {meta_info['doc_title']} updated in {len(new_chunks)} chunks")
    return True


async def main(parser: DocxParser, filepaths: List[str], collection, api_version):
    CLIENT_SESSION_WRAPPER.coreml_session = ClientSession(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}"
    )
    CLIENT_SESSION_WRAPPER.general_session = ClientSession()

    results = await asyncio.gather(*[process_file(fp, parser, collection, api_version) for fp in filepaths])
    # todo: repeat unsuccessful
    print(results)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="path to folder with docx  files")
    parser.add_argument("--api_version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    collection_name = "internal"
    vendor = "oneclickcx"
    organization = hash_string(vendor)

    utility.drop_collection(f"{vendor}_{organization}_{collection_name}")

    collection = MILVUS_DB.get_or_create_collection(f"{vendor}_{organization}_{collection_name}")
    docx_parser = DocxParser(1024)

    all_docs = glob.glob(osp.join(args.source, "*.docx"))
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(docx_parser, all_docs, collection, args.api_version))
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()

    # https://stackoverflow.com/questions/47169474/parallel-asynchronous-io-in-pythons-coroutines
