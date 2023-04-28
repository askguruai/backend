import os
import sys

sys.path.insert(1, os.getcwd())
import traceback

import hashlib
import logging
from argparse import ArgumentParser

from tqdm import tqdm

from parsers.markdown_parser import MarkdownParser
from utils.errors import CoreMLError
from utils.ml_requests import get_embeddings
from pymilvus import (
    utility as m_utility,
    Collection,
)
from milvus_db.utils import get_or_create_collection


def process_single_file(milvus_collection: Collection,
                        path: str,
                        subcollection: str, parser: MarkdownParser, api_version: str):
    chunks, title = parser.process_file(path)
    new_chunks = []
    id_hashes = []
    for chunk in chunks:
        text_hash = hashlib.sha256(chunk.encode()).hexdigest()[:24]
        res = milvus_collection.query(
            expr=f"id_hash==\"{text_hash}\"",
            offset=0,
            limit=1,
            output_fields=["id_hash"],
            consistency_level="Strong"
        )
        if len(res) == 0:
            # there is no such document yet, inserting
            id_hashes.append(text_hash)
            new_chunks.append(chunk)
    if len(new_chunks) == 0:
        # nothing to for this document
        return True
    try:
        embeddings = get_embeddings(new_chunks, api_version=api_version)
    except CoreMLError:
        traceback.print_exc()
        print("Failed")
        return False
    assert len(embeddings) == len(new_chunks)

    # todo: maybe insert data in even bigger chunks because indexing idk
    data = [
        id_hashes,
        new_chunks,
        embeddings,
        [title] * len(id_hashes),
        [f"{subcollection}.com/help/{title.lower().replace(' ', '-')}"] * len(id_hashes)
    ]
    milvus_collection.insert(data)
    logging.info(f"Document {title} in {len(id_hashes)} chunks inserted in the database")
    print(f"Document {title} in {len(id_hashes)} chunks inserted in the database")
    return True


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str)
    parser.add_argument("--api_version", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()

    subcollection_name = args.source_dir.split("_")[0].split("-")[1]
    vendor = "livechat"
    organization_id = "f1ac8408-27b2-465e-89c6-b8708bfc262c"
    org_hash = hashlib.sha256(organization_id.encode()).hexdigest()[:24]

    collection_name = f"{vendor}_{org_hash}_{subcollection_name}"
    m_collection = get_or_create_collection(collection_name)

    md_parser = MarkdownParser(2048)
    docs = os.listdir(args.source_dir)
    for doc in tqdm(docs):
        if doc.endswith(".md") or doc.endswith(".markdownd"):
            process_ok = False
            while not process_ok:
                process_ok = process_single_file(
                    path=os.path.join(args.source_dir, doc),
                    subcollection=subcollection_name,
                    parser=md_parser,
                    api_version=args.api_version,
                    milvus_collection=m_collection
                )
    # todo: takes long -- investigate!
    # m_utility.flush_all()
