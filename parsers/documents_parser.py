import asyncio
import os.path as osp
from collections import deque
from datetime import datetime
from typing import List, Tuple
from urllib.parse import urljoin

import html2text
import tiktoken
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from fastapi import HTTPException, status
from langdetect import detect as language_detect
from loguru import logger
from starlette.datastructures import UploadFile as StarletteUploadFile

from parsers.docx_parser_ import DocxParser
from parsers.markdown_parser import MarkdownParser
from parsers.pdf_parser import PdfParser
from utils import GRIDFS, TRANSLATE_CLIENT
from utils.misc import int_list_encode
from utils.schemas import Chat, Doc, DocumentMetadata
from utils.tokenize_ import doc_to_chunks

DOCX_PARSER = DocxParser(1024)
PDF_PARSER = PdfParser(1024)
MD_PARSER = MarkdownParser(1024)


class DocumentsParser:
    def __init__(self, chunk_size: int, tokenizer_name: str):
        self.chunk_size = chunk_size
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_images = True

    def process_document(self, document: Chat | Doc, metadata: DocumentMetadata) -> Tuple[List[str], dict]:
        if isinstance(document, Doc):
            meta = {
                "doc_id": metadata.id,
                "doc_title": metadata.title,
                "timestamp": int(metadata.timestamp)
                if metadata.timestamp is not None
                else int(datetime.now().timestamp()),
                "doc_summary": metadata.summary if metadata.summary is not None else "",
                "security_groups": int_list_encode(metadata.security_groups)
                if metadata.security_groups is not None
                else 2**63 - 1,
                "url": metadata.url if metadata.url else "",
                "source_language": None,
            }
            if metadata.project_to_en:
                detection_result = TRANSLATE_CLIENT.detect_language(document.content[:512])
                document_language = detection_result["language"]
                if document_language != "en":
                    trans_result = TRANSLATE_CLIENT.translate(document.content, target_language="en", format_="text")
                    content = trans_result["translatedText"]
                    meta["source_language"] = trans_result["detectedSourceLanguage"]
                else:
                    content = document.content
            else:
                content = document.content
            chunks = doc_to_chunks(content, meta["doc_title"], meta["doc_summary"])

        elif isinstance(document, Chat):
            meta = {
                "doc_id": metadata.id,
                "doc_title": f"{document.user.name}::{document.user.id}",
                "doc_summary": metadata.summary if metadata.summary is not None else "",
                "timestamp": int(metadata.timestamp)
                if metadata.timestamp is not None
                else int(datetime.now().timestamp()),
                "security_groups": int_list_encode(metadata.security_groups),
                "url": metadata.url if metadata.url else "",
                "source_language": None,
            }
            if len(document.history) == 0:
                return None, None, None
            text_lines = [f"{message.role}: {message.content}" for message in document.history]
            raw_lines = [message.content for message in document.history]
            if metadata.project_to_en:
                detection_result = TRANSLATE_CLIENT.detect_language("\n".join(raw_lines[:5]))
                document_language = detection_result["language"]
                if document_language != "en":
                    trans_result = TRANSLATE_CLIENT.translate(
                        raw_lines, target_language="en", format_="text", model="nmt"
                    )
                    translated_lines = [trans_line["translatedText"] for trans_line in trans_result]
                    text_lines = [f"{ent[0].role}: {ent[1]}" for ent in zip(document.history, translated_lines)]
                    meta["source_language"] = trans_result[0]["detectedSourceLanguage"]
            content = "\n".join(text_lines)
            chunks = self.chat_to_chunks(text_lines)
        return chunks, meta, content

    async def process_link(
        self,
        session: ClientSession,
        link: str,
        root_link: str,
        queue: deque,
        visited: set,
        ignore_urls: bool = True,
        allow_redirects: bool = True,
    ) -> Doc:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
        }
        try:
            async with session.get(link, headers=headers, allow_redirects=allow_redirects) as response:
                page_content = await response.text()
        except Exception as e:
            logger.error(f"Error while downloading {link}: {e}")
            return None

        if page_content and page_content.count("<html") > 0:
            soup = BeautifulSoup(page_content, "html.parser")
            for a in soup.find_all(href=True):
                url = urljoin(link, a["href"]).split("#")[0].split("?")[0].split(" ")[0]
                is_file = url.split("/")[-1].count(".") > 0
                if (
                    url not in visited
                    and url.startswith(root_link)
                    and "wp-json" not in url
                    and (not is_file or url.endswith(".html") or url.endswith(".htm"))
                    and url + "/" not in visited
                    and url[:-1] not in visited
                    and "<" not in url
                ):
                    queue.append(url)
                    visited.add(url)
            title = soup.find("title").text if (soup.find("title") and soup.find("title").text) else link
            if ignore_urls != self.converter.ignore_links:
                self.converter.ignore_links = ignore_urls
                content = self.converter.handle(page_content)
                self.converter.ignore_links = not ignore_urls
            else:
                content = self.converter.handle(page_content)
            if not content:
                logger.error(f"Empty content on {link}")
                return None
            return Doc(id=link, title=title, summary=title, content=content)

        return None

    async def link_to_docs(
        self, root_link: str, max_depth: int = 50, max_total_docs: int = 500, ignore_urls: bool = True
    ) -> List[Doc]:
        if root_link[-1] != "/":
            root_link += "/"
        queue, visited, depth = deque([root_link]), set([root_link]), 0
        docs = []

        async with ClientSession() as session:
            while queue and depth < max_depth and len(docs) < max_total_docs:
                tasks = []
                logger.info(
                    f"Depth: {depth} / {max_depth}, total: {len(docs)} / {max_total_docs}, queue size: {len(queue)}, link: {root_link}"
                )
                for _ in range(len(queue)):
                    tasks.append(
                        self.process_link(
                            session,
                            queue.popleft(),
                            root_link,
                            queue,
                            visited,
                            ignore_urls=ignore_urls,
                        )
                    )

                results = await asyncio.gather(*tasks)

                for result in results:
                    if result:
                        docs.append(result)

                depth += 1

        logger.info(f"Found {len(docs)} documents on {root_link}")
        return docs[:max_total_docs]

    async def raw_to_doc(
        self, file: StarletteUploadFile, vendor: str, organization: str, collection: str, doc_id: str
    ) -> Doc:
        contents = await file.read()
        name, format = osp.splitext(file.filename)
        if format == ".pdf":
            text = PDF_PARSER.stream2text(stream=contents)
        elif format == ".docx":
            text = DOCX_PARSER.stream2text(stream=contents)
        elif format == ".md":
            text = MD_PARSER.stream2text(stream=contents)
        else:
            msg = f"Uploading files of type {format} is not supported. Allowed types: pdf, docx, and md"
            logger.error(msg)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=msg,
            )

        doc = Doc(content=text)

        filename = f"{vendor}_{organization}_{collection}_{doc_id}"
        res = GRIDFS.find_one({"filename": filename})
        if res:
            GRIDFS.delete(res._id)
            logger.info(f"Deleted file {filename} from GridFS")
        GRIDFS.put(
            contents,
            filename=filename,
            content_type=file.content_type,
        )
        return doc

    def chat_to_chunks(self, text_lines: List[str]) -> List[str]:
        return doc_to_chunks(content="---***---".join(text_lines), splitter="---***---", overlapping_lines=10)
