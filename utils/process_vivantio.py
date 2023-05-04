import os
import sys

sys.path.insert(1, os.getcwd())

import asyncio
import hashlib
import json
import logging
import pickle
from argparse import ArgumentParser

from bson.binary import Binary
from bson.objectid import ObjectId
from pymilvus import Collection
from tqdm import tqdm

from parsers.html_parser import VivantioHTMLParser
from utils import CONFIG, DB, MILVUS_DB, hash_string, ml_requests
from utils.errors import CoreMLError


def process_single_file(
    document: dict, collection: Collection, parser: VivantioHTMLParser, api_version: str
) -> bool:
    chunks, meta_info = parser.process_document(document)
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

    embeddings = ml_requests.get_embeddings_sync(new_chunks, api_version=api_version)

    data = [
        new_chunks_hashes,
        [meta_info["doc_id"]] * len(new_chunks),
        new_chunks,
        embeddings,
        [meta_info["doc_title"]] * len(new_chunks),
        [""] * len(new_chunks),
    ]

    collection.insert(data)
    logging.info(f"Document {meta_info['doc_title']} updated in {len(new_chunks)} chunks")
    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="path to a processed .json file")
    parser.add_argument("--api_version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    collection_name = "internal"
    vendor = "vivantio"
    organization = hash_string("vivantio")

    collection = MILVUS_DB.get_or_create_collection(
        f"{vendor}_{organization}_{collection_name}"
    )
    print(f"Currently {collection.num_entities} entities")
    h_parser = VivantioHTMLParser(2000)
    with open(args.source, "rt") as f:
        data = json.load(f)

    all_docs = data["Results"]
    for doc in tqdm(all_docs):
        process_ok = False
        while not process_ok:
            process_ok = process_single_file(
                doc,
                collection,
                h_parser,
                api_version=args.api_version,
            )
