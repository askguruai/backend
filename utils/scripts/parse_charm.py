import asyncio
import os
import re
import sys
from datetime import datetime
from typing import List, Tuple
from urllib.parse import unquote

import html2text
import tiktoken
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

sys.path.insert(1, os.getcwd())

from utils import CLIENT_SESSION_WRAPPER, CONFIG, MILVUS_DB, hash_string, ml_requests

MAX_TOKENS_PER_CHUNK = 1024
TARGET_FOLDER = "./data/charmli/2013 Ford Truck F 150 4WD V8-6.2L/"
CORE_URL = "https://charm.li/Ford%20Truck/2013/F%20150%204WD%20V8-6.2L/"

VENDOR = "askguru"
ORGANIZATION = "askguru"
COLLECTION_NAME = "charmli"


def extract_info(core_url: str, target_folder: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    def append_to_lists(link, path, title, content):
        if len(enc.encode(content)) > MAX_TOKENS_PER_CHUNK:
            logger.warning(f"Document {link} has {len(enc.encode(content))} tokens and length {len(content)}")

        # if link == "https://charm.li/Ford%20Truck/2013/F%20150%204WD%20V8-6.2L/Repair%20and%20Diagnosis/Engine%2C%20Cooling%20and%20Exhaust/Cooling%20System/Service%20and%20Repair/":
        #     logger.info(content)
        # logger.info(f"{link}\n{content}")
        links.append(link)
        paths.append(path)
        titles.append(title)
        contents.append(content)

    enc = tiktoken.get_encoding("cl100k_base")
    n_docs_bigger_than_max_tokens, n_docs = 0, 0

    converter = html2text.HTML2Text()
    converter.ignore_links = True

    links, paths, titles, contents = [], [], [], []

    for root, dirs, files in tqdm(os.walk(target_folder, topdown=True)):
        if not dirs and files == ["index.html"]:
            with open(os.path.join(root, "index.html"), "r") as f:
                html = f.read()
            main_div = BeautifulSoup(html, "html.parser").find("div", {"class": "main"})
            # html to md does not takes into account
            # class "indent-x" on sublists, e.g.
            # https://charm.li/Ford%20Truck/2013/F%20150%204WD%20V8-6.2L/Repair%20and%20Diagnosis/Relays%20and%20Modules/Relays%20and%20Modules%20-%20Restraints%20and%20Safety%20Systems/Air%20Bag%20Control%20Module/Service%20and%20Repair/
            content = re.sub(r'!\[.*?\]\(.*?\)', '', converter.handle(str(main_div))).strip()

            # docs with only title, e.g some dealer
            # letters or attachments-only
            if content.count("\n") < 1:
                continue

            title, *content = content.split("\n")
            title, content = title[2:], "\n".join(content).strip()
            link = core_url + root[len(target_folder) :] + "/"
            path = " > ".join([unquote(category) for category in link.split("/")[3:-1]])

            if len(enc.encode(f"---\npath: {path}\ntitle: {title}\n---\n\n{content}")) > MAX_TOKENS_PER_CHUNK:
                n_docs_bigger_than_max_tokens += 1
                part = 0
                current_content = f"---\npart: {part}\npath: {path}\ntitle: {title}\n---\n\n"
                # TODO: split by lines which are bolded
                # because they are the titles of the sections
                for line in content.split("\n"):
                    if len(enc.encode(current_content + line + "\n")) > MAX_TOKENS_PER_CHUNK:
                        append_to_lists(link, path, title, current_content)
                        part += 1
                        current_content = f"---\npart: {part}\npath: {path}\ntitle: {title}\n---\n\n"
                    current_content += line + "\n"
                append_to_lists(link, path, title, current_content)
            else:
                content = f"---\npath: {path}\ntitle: {title}\n---\n\n{content}"
                append_to_lists(link, path, title, content)

            n_docs += 1

    logger.info(f"Number of chunks bigger than {MAX_TOKENS_PER_CHUNK} tokens: {n_docs_bigger_than_max_tokens}")
    logger.info(f"Number of docs: {n_docs}")
    return links, paths, titles, contents


async def main():
    links, paths, titles, contents = extract_info(CORE_URL, TARGET_FOLDER)
    logger.info(f"Total number of chunks: {len(contents)}")

    CLIENT_SESSION_WRAPPER.coreml_session = ClientSession(
        f"http://{CONFIG['coreml']['host']}:{CONFIG['coreml']['port']}"
    )
    embeddings = await ml_requests.get_embeddings(contents, api_version="v1")

    collection = MILVUS_DB.get_or_create_collection(f"{VENDOR}_{hash_string(ORGANIZATION)}_{COLLECTION_NAME}")

    data = [
        list(map(hash_string, contents)),
        links,
        contents,
        embeddings,
        titles,
        paths,
        [int(datetime.now().timestamp())] * len(contents),
        [2**63 - 1] * len(contents),
    ]

    collection.insert(data)

    CLIENT_SESSION_WRAPPER.coreml_session.close()


if __name__ == "__main__":
    asyncio.run(main())
