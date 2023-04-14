import os
import sys

sys.path.insert(1, os.getcwd())

import hashlib
import logging
import json
import pickle
from argparse import ArgumentParser

from bson.binary import Binary
from bson.objectid import ObjectId
from tqdm import tqdm

from utils import DB
from utils.errors import CoreMLError
from utils.ml_requests import get_embeddings
from parsers.html_parser import VivantioHTMLParser


def process_single_file(document: dict, collection: str, parser: VivantioHTMLParser, api_version: str) -> bool:
    chunks, meta_info = parser.process_document(document)
    if len(chunks) == 0:
        return True  # nothing to do
    try:
        embeddings = get_embeddings(chunks, api_version=api_version)
    except CoreMLError:
        print("Failed")
        return False
    assert len(embeddings) == len(chunks)
    for i, pair in enumerate(zip(chunks, embeddings)):
        chunk, emb = pair
        text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:24]
        document = DB[f"{api_version}.{collection}"].find_one(
            {"_id": ObjectId(text_hash)}
        )
        if not document:
            document = {
                "_id": ObjectId(text_hash),
                "doc_title": meta_info["title"],
                "link": meta_info["link"],
                "doc_id": meta_info["id"],
                "chunk": chunk,
                "embedding": Binary(pickle.dumps(emb)),
            }
            DB[f"{api_version}.{collection}"].insert_one(document)
            logging.info(f"Document {meta_info['title']} chunk {i} inserted in the database")
    return True


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="path to a processed .json file")
    parser.add_argument("--api_version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--cname", type=str, help="collection name (will be cnnected with api version")
    args = parser.parse_args()

    h_parser = VivantioHTMLParser(2000)
    with open(args.source, "rt") as f:
        data = json.load(f)

    all_docs = data["Results"]
    for doc in tqdm(all_docs):
        process_ok = False
        while not process_ok:
            process_ok = process_single_file(
                doc,
                args.cname,
                h_parser,
                api_version=args.api_version,
            )

