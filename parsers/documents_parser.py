import asyncio
from collections import deque
from datetime import datetime
from typing import List, Tuple
from urllib.parse import urljoin

import html2text
import requests
import tiktoken
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from loguru import logger

from utils import hash_string
from utils.misc import int_list_encode
from utils.schemas import Chat, Doc


class DocumentsParser:
    def __init__(self, chunk_size: int, tokenizer_name: str):
        self.chunk_size = chunk_size
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_images = True

    def process_document(self, document: Chat | Doc) -> Tuple[List[str], dict]:
        if isinstance(document, Doc):
            meta = {
                "doc_id": document.id if document.id is not None else hash_string(document.content),
                "doc_title": document.title if document.title is not None else "",
                "doc_summary": document.summary if document.summary is not None else "",
                "timestamp": int(document.timestamp)
                if document.timestamp is not None
                else int(datetime.now().timestamp()),
                "security_groups": int_list_encode(document.security_groups)
                if document.security_groups is not None
                else 2**63 - 1,
            }
            chunks = self.doc_to_chunks(document.content, meta["doc_title"], meta["doc_summary"])
        elif isinstance(document, Chat):
            meta = {
                "doc_id": document.id,
                "doc_title": f"{document.user.name}::{document.user.id}",
                "doc_summary": "",
                "timestamp": int(document.timestamp),
                "security_groups": int_list_encode(document.security_groups),
            }
            text_lines = [f"{message.role}: {message.content}" for message in document.history]
            chunks = self.chat_to_chunks(text_lines)
        return chunks, meta

    async def process_link(self, session: ClientSession, link: str, root_link: str, queue: deque, visited: set) -> Doc:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        }
        try:
            async with session.get(link, headers=headers) as response:
                page_content = await response.text()
        except Exception as e:
            logger.error(f"Error while downloading {link}: {e}")
            return None

        if page_content:
            soup = BeautifulSoup(page_content, "html.parser")
            for a in soup.find_all(href=True):
                url = urljoin(link, a["href"]).split("#")[0].split("?")[0]
                is_file = url.split("/")[-1].count(".") > 0
                if (
                    url not in visited
                    and "wp-json" not in url
                    and url.startswith(root_link)
                    and (not is_file or url.endswith(".html"))
                ):
                    queue.append(url)
                    visited.add(url)
            title = soup.find("title").text if (soup.find("title") and soup.find("title").text) else link
            content = self.converter.handle(page_content)
            return Doc(id=link, title=title, summary=title, content=content)

        return None

    async def link_to_docs(self, root_link: str) -> List[Doc]:
        if root_link[-1] != "/":
            root_link += "/"
        queue, visited, depth = deque([root_link]), set([root_link]), 0
        docs = []

        async with ClientSession() as session:
            while queue:
                tasks = []
                logger.info(f"Depth: {depth}, queue size: {len(queue)}, link: {root_link}")
                for _ in range(len(queue)):
                    tasks.append(self.process_link(session, queue.popleft(), root_link, queue, visited))

                results = await asyncio.gather(*tasks)

                for result in results:
                    if result:
                        docs.append(result)

                depth += 1

        logger.info(f"Found {len(docs)} documents on {root_link}")
        return docs

    def doc_to_chunks(self, content: str, title: str = "", summary: str = "") -> List[str]:
        chunks = []

        # TODO: omit title because it is already in summary

        current_content, part = f"{summary}\n\n", 0
        # TODO: split by lines which are bolded (in case of cars)
        # because they are the titles of the sections
        for line in content.split("\n"):
            if len(self.enc.encode(current_content + line + "\n")) > self.chunk_size:
                chunks.append(current_content.strip())
                current_content = f"{summary}\n\n"
                part += 1

            line_tokens = self.enc.encode(line)
            if len(line_tokens) > self.chunk_size:
                line_token_parts = [
                    line_tokens[i : i + self.chunk_size] for i in range(0, len(line_tokens), self.chunk_size)
                ]

                for line_token_part in line_token_parts:
                    line_part = self.enc.decode(line_token_part)
                    if len(self.enc.encode(current_content + line_part + "\n")) > self.chunk_size:
                        chunks.append(current_content.strip())
                        current_content = f"{summary}\n\n"
                    current_content += line_part + "\n"

            else:
                current_content += line + "\n"

        chunks.append(current_content.strip())
        return chunks

    def chat_to_chunks(self, text_lines: List[str]) -> List[str]:
        chunks = []
        cur_chunk = ""
        while len(text_lines) > 0:
            line = text_lines.pop(0)
            if len(cur_chunk) + len(line) < self.chunk_size:
                cur_chunk += f"\n{line}"
            else:
                chunk = cur_chunk.strip()
                if len(chunks) > 0:
                    last_line = chunks[-1].rsplit("\n", maxsplit=1)[1]
                    chunk = f"{last_line}\n{chunk}"
                chunks.append(chunk)
                cur_chunk = line
        if cur_chunk != "":
            chunk = cur_chunk.strip()
            if len(chunks) > 0:
                last_line = chunks[-1].rsplit("\n", maxsplit=1)[1]
                chunk = f"{last_line}\n{chunk}"
            chunks.append(chunk)
        return chunks
