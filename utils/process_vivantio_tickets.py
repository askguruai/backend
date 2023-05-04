import os
import sys

sys.path.insert(1, os.getcwd())

import hashlib
import json
import logging
import pickle
from argparse import ArgumentParser

from bson.binary import Binary
from bson.objectid import ObjectId
from tqdm import tqdm

from parsers.html_parser import VivantioHTMLParser
from utils import DB, hash_string
from utils.errors import CoreMLError
from utils.ml_requests import get_embeddings


def process_single_file(document: dict, collection: str, api_version: str) -> bool:
    if "summary" not in document:
        # nothing to do
        return True
    summary = document["summary"]
    if "raw" in summary:
        chunk = summary["raw"][:1024]
        short_description = document["hdctitle"]
    elif "problem" in summary and "solution" in summary:
        chunk = f"Problem:  {str(summary['problem'])}"
        if summary["solution"] is not None:
            chunk += f"\nSolution:  {str(summary['solution'])}"
        short_description = str(summary["problem"])
    else:
        print(f"Ticket {document['idhdcall']} has malformed summary")
        return True
    if "hdctitle" in document:
        chunk = f"{document['hdctitle']}\n{chunk}"
    if "hdcatLineage" in document:
        chunk = f"{document['hdcatLineage']}\n{chunk}"
    text_hash = hash_string(chunk)
    db_document = DB[f"{api_version}.collections.vivantio.vivantio.{collection}"].find_one(
        {"_id": ObjectId(text_hash)}
    )
    if not db_document:
        try:
            embedding = get_embeddings(chunk, api_version=api_version)[0]
        except CoreMLError:
            print("Failed")
            return False
        db_document = {
            "_id": ObjectId(text_hash),
            "doc_title": document["hdctitle"],
            "link": f"https://vivantio.flex.vivantio.com/item/Ticket/{document['idhdcall']}",
            "doc_id": str(document["idhdcall"]),
            "chunk": chunk,
            "description": short_description,
            "embedding": Binary(pickle.dumps(embedding)),
        }
        DB[f"{api_version}.collections.vivantio.vivantio.{collection}"].insert_one(db_document)
        logging.info(f"Document {document['hdctitle']} inserted in the database")
    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="path to a processed .json file")
    parser.add_argument("--api_version", choices=["v1", "v2"], default="v1")
    parser.add_argument(
        "--cname", type=str, help="collection name (will be cnnected with api version"
    )
    args = parser.parse_args()

    with open(args.source, "rt") as f:
        data = json.load(f)

    for doc in tqdm(data):
        process_ok = False
        while not process_ok:
            process_ok = process_single_file(
                doc,
                args.cname,
                api_version=args.api_version,
            )
