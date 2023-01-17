import numpy as np
import requests
from utils import CONFIG, DB
import html2text
from requests.auth import HTTPBasicAuth
from typing import List, Tuple
from bson.objectid import ObjectId
from bson.binary import Binary
import pickle
from parsers.general_parser import GeneralParser as GP
from handlers.general_handler import GeneralHandler as GH
from utils.ml_requests import get_embeddings
from utils.api import ConfluenceSearchRequest
from textblob import TextBlob

DOMAIN = 'testaskai.atlassian.net'
TOKEN = 'kCYEAtKobeBsEcc82GCzD51E'
EMAIL = "spgorbatiuk@gmail.com"


def get_search_result_ids(query: str) -> List[str]:
    auth = HTTPBasicAuth(EMAIL, TOKEN)
    nouns = TextBlob(query).noun_phrases
    noun_query = [f"(text~\"{noun}\")" for noun in nouns]
    noun_query = " or ".join(noun_query)
    cql_query = f"(type=page) and ({noun_query})"
    query = {
        'cql': cql_query
    }
    response = requests.get(
        f"https://{DOMAIN}/wiki/rest/api/search",
        headers={"Accept": "application/json"},
        params=query,
        auth=auth
    )
    response_data = response.json()
    if "results" not in response_data:
        return []
    page_ids = []
    for res in response_data["results"]:
        page_ids.append(res["content"]["id"])
    return page_ids


def search_request_handler(request: ConfluenceSearchRequest):
    query = request.query
    search_ids = get_search_result_ids(query)
    for page_id in search_ids:
        print(f"Page {page_id}")
        chunks, embeddings = get_page_by_id(page_id)
        print(chunks[0])


def get_page_contents(page_id: str) -> Tuple[str, str]:
    auth = HTTPBasicAuth(EMAIL, TOKEN)

    page_contents = requests.get(f"https://{DOMAIN}/wiki/rest/api/content/{page_id}?expand=body.storage",
                                 headers={
                                     'Accept': 'application/json',
                                 },
                                 auth=auth)
    response = page_contents.json()
    page_title = response["title"]
    page_text = html2text.html2text(response['body']['storage']['value'])
    return page_title, page_text


def get_page_by_id(page_id):
    document = DB[DOMAIN].find_one(
        {"_id": page_id}
    )
    if document is None:
        chunks, embeddings = process_confluence_page(page_id=page_id)
    else:
        page_title, page_text = get_page_contents(page_id)
        content_hash = GH.get_hash(page_text)
        stored_hash = document["hash"]
        if content_hash != stored_hash:  # page content has changed
            chunks, embeddings = process_confluence_page(page_id=page_id)
        else:
            chunks, embeddings = document["chunks"], pickle.loads(document["embeddings"])

    return chunks, embeddings


def process_confluence_page(page_id: str) -> Tuple[List[str], List[np.ndarray]]:
    page_title, page_text = get_page_contents(page_id)
    content_hash = GH.get_hash(page_text)
    sentences = GP.text_to_sentences(page_text)
    chunks = GP.chunkise_sentences(sentences, chunk_size=int(CONFIG["handlers"]["chunk_size"]))
    embeddings = get_embeddings(chunks)
    document = {
        "_id": page_id,
        "hash": content_hash,
        "title": page_title,
        "text": page_text,
        "chunks": chunks,
        "embeddings": Binary(pickle.dumps(embeddings)),
    }
    DB[DOMAIN].insert_one(document)
    return chunks, embeddings

