import sys
import os, os.path as osp
import json
from utils import hash_string, CONFIG, MILVUS_DB, ml_requests
from utils.tokenize_ import doc_to_chunks
from argparse import ArgumentParser
import glob
import asyncio
from typing import List
from loguru import logger

sys.path.insert(1, os.getcwd())

from parsers import DocxParser

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
    "How to use touchpad.docx": "https://docs.google.com/document/d/1qQPO0ttjb8w8dOkCxAYYy1Kn-6Y6jbqS/"
}

async def process_file(filepath: str, parser: DocxParser, collection, api_version):
    filename = osp.split(filepath)[1]
    doc_link = DOCS_N_LINKS[filename]
    chunks, meta_info = parser.process_file(filepath)
    if len(chunks) == 0:
        return True  # nothing to do
    doc_id = meta_info["doc_id"]
    existing_chunks = collection.query(
        expr=f'doc_id=="{doc_id}"',
        offset=0,
        limit=10000,
        output_fields=["chunk_hash"],
        consistency_level="Strong",
    )
    existing_chunks = set((hit["chunk_hash"] for hit in existing_chunks))
    new_chunks_hashes = []
    new_chunks = []
    for chunk in chunks:
        text_hash = hash_string(chunk)
        if text_hash in existing_chunks:
            existing_chunks.remove(text_hash)
        else:
            new_chunks.append(chunk)
            new_chunks_hashes.append(text_hash)
    # dropping outdated chunks
    existing_chunks = [f'"{ch}"' for ch in existing_chunks]
    collection.delete(f"chunk_hash in [{','.join(existing_chunks)}]")

    if len(new_chunks) == 0:
        # everyting is already in the database
        return True
    embeddings = await ml_requests.get_embeddings(new_chunks)

    data = [
        new_chunks_hashes,
        [meta_info["doc_id"]] * len(new_chunks),
        new_chunks,
        embeddings,
        [meta_info["doc_title"]] * len(new_chunks),
        [""] * len(new_chunks),
        [0] * len(new_chunks),
        [2 ** 63 - 1] * len(new_chunks)
    ]
    collection.insert(data)
    logger.info(f"Document {meta_info['doc_title']} updated in {len(new_chunks)} chunks")
    return True


async def main(parser:DocxParser, filepaths: List[str], collection, api_version):
    results = await asyncio.gather(
        process_file(filepaths[0], parser, collection, api_version),
        process_file(filepaths[1], parser, collection, api_version),
        process_file(filepaths[2], parser, collection, api_version)
    )
    print(results)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="path to folder with docx  files")
    parser.add_argument("--api_version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    collection_name = "internal"
    vendor = "oneclickcx"
    organization = hash_string(vendor)

    collection = MILVUS_DB.get_or_create_collection(f"{vendor}_{organization}_{collection_name}")
    docx_parser = DocxParser(1024)

    all_docs = glob.glob(osp.join(args.source, "*.docx"))
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(docx_parser, all_docs, collection, api_version))
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()

    # https://stackoverflow.com/questions/47169474/parallel-asynchronous-io-in-pythons-coroutines
    