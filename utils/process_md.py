import os
import sys

sys.path.insert(1, os.getcwd())

import hashlib
import logging
import pickle
from argparse import ArgumentParser

from bson.binary import Binary
from bson.objectid import ObjectId
from tqdm import tqdm

from parsers.markdown_parser import MarkdownParser
from utils import DB
from utils.errors import CoreMLError
from utils.ml_requests import get_embeddings


def process_single_file(path, collection, parser: MarkdownParser, api_version: str):
    chunks, title = parser.process_file(path)
    try:
        embeddings = get_embeddings(chunks, api_version=api_version)
    except CoreMLError:
        print("Failed")
        return False
    assert len(embeddings) == len(chunks)
    for i, pair in enumerate(zip(chunks, embeddings)):
        chunk, emb = pair
        text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:24]
        document = DB[f"{api_version}.collections.livechat.{collection}"].find_one(
            {"_id": ObjectId(text_hash)}
        )
        if not document:
            document = {
                "_id": ObjectId(text_hash),
                "doc_title": title,
                "chunk": chunk,
                "embedding": Binary(pickle.dumps(emb)),
            }
            DB[f"{api_version}.collections.livechat.{collection}"].insert_one(document)
            logging.info(f"Document {title} chunk {i} inserted in the database")
    return True


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str)
    parser.add_argument("--api_version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    collection_name = args.source_dir.split("_")[0].split("-")[1]
    md_parser = MarkdownParser(1024)
    docs = os.listdir(args.source_dir)
    for doc in tqdm(docs):
        if doc.endswith(".md") or doc.endswith(".markdownd"):
            process_ok = False
            while not process_ok:
                process_ok = process_single_file(
                    os.path.join(args.source_dir, doc),
                    collection_name,
                    md_parser,
                    api_version=args.api_version,
                )
