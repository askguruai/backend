import asyncio
import os.path as osp
import re
from collections import deque
from datetime import datetime
from typing import List, Tuple
from urllib.parse import urljoin

import html2text
import tiktoken
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from fastapi import HTTPException, status
from loguru import logger
from starlette.datastructures import UploadFile as StarletteUploadFile

from parsers.docx_parser_ import DocxParser
from parsers.markdown_parser import MarkdownParser
from parsers.pdf_parser import PdfParser
from utils import AWS_TRANSLATE_CLIENT, GRIDFS, full_collection_name
from utils.misc import int_list_encode
from utils.schemas import Chat, Doc, DocumentMetadata
from utils.tokenize_ import doc_to_chunks

# from cache import AsyncTTL

DOCX_PARSER = DocxParser(1024)
PDF_PARSER = PdfParser(1024)
MD_PARSER = MarkdownParser(1024)


class DocumentsParser:
    def __init__(self, chunk_size: int, tokenizer_name: str):
        self.chunk_size = chunk_size
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_images = True
        self.sitemap_pattern = r"sitemap.*\.xml"

    # @AsyncTTL(time_to_live=60, maxsize=4096)
    async def get_page_content(self, session: ClientSession, link: str, allow_redirects: bool = True) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
        }
        link_clickhelp_adjusted = link.replace("articles/#!", "article/")
        try:
            async with session.get(
                link_clickhelp_adjusted, headers=headers, allow_redirects=allow_redirects, timeout=600
            ) as response:
                page_content = await response.text()
        except Exception as e:
            logger.error(f"Error while downloading {link_clickhelp_adjusted}: {e}")
            return None

        return page_content

    async def extract_urls(
        self,
        session: ClientSession,
        link: str,
        allow_redirects: bool = True,
        root_url: str = "",
        visited: set = set(),
    ) -> List[str]:
        page_content = await self.get_page_content(session=session, link=link, allow_redirects=allow_redirects)

        if "<html" in page_content:
            soup = BeautifulSoup(page_content, "html.parser")
            link_candidates = [
                urljoin(link, a["href"]).split("#")[0].split("?")[0].split(" ")[0] for a in soup.find_all(href=True)
            ]
        elif "<?xml" in page_content:
            soup = BeautifulSoup(page_content, "xml")
            link_candidates = [a.text.strip() for a in soup.find_all("loc")]
        else:
            logger.warning(f"Unknown content format on link: {link}")
            return []

        links_found = []
        for url in link_candidates:
            is_file = url.split("/")[-1].count(".") > 0
            if (
                url not in visited
                and url.startswith(root_url)
                and "wp-json" not in url
                and (not is_file or url.endswith(".html") or url.endswith(".htm") or url.endswith(".xml"))
                and url + "/" not in visited
                and url[:-1] not in visited
                and "<" not in url
                # and "/de" not in url
                # and "/de-ch" not in url
                # and "/fr" not in url
            ):
                visited.add(url)
                links_found.append(url)

        return links_found

    async def process_link(
        self,
        session: ClientSession,
        link: str,
        allow_redirects: bool = True,
    ) -> Tuple[Doc, DocumentMetadata]:
        page_content = await self.get_page_content(session=session, link=link, allow_redirects=allow_redirects)
        soup = BeautifulSoup(page_content, "html.parser")

        for each in ["header", "footer"]:
            s = soup.find(each)
            if s:
                s.extract()
        div_to_remove = soup.find_all("div", class_="kb-footer")
        if div_to_remove:
            div_to_remove[0].extract()

        page_content = str(soup)

        title = soup.find("title").text if (soup.find("title") and soup.find("title").text) else link
        content = self.converter.handle(page_content)

        if "redirecting" in title.lower():
            logger.warning(f"Redirecting page on {link}")
            return None, None, None

        # if not content:
        #     logger.error(f"Empty content on {link}")
        #     return None

        return Doc(content=content), DocumentMetadata(id=link, title=title, url=link, project_to_en=False), page_content

    async def traverse_page(
        self, root_link: str, max_depth: int = 50, max_total_docs: int = 500
    ) -> List[Tuple[Doc, DocumentMetadata, str]]:
        if root_link[-1] != "/":
            root_link += "/"

        queue, visited, depth = deque([root_link]), set([root_link]), 0
        docs = []

        async with ClientSession() as session:
            while queue and depth < max_depth and len(docs) < max_total_docs:
                logger.info(
                    f"Depth: {depth} / {max_depth}, total: {len(docs)} / {max_total_docs}, queue size: {len(queue)}, link: {root_link}"
                )
                tasks_process_link = []
                tasks_extract_urls = []
                for _ in range(len(queue)):
                    url = queue.popleft()
                    tasks_process_link.append(self.process_link(session=session, link=url))
                    tasks_extract_urls.append(
                        self.extract_urls(
                            session=session,
                            link=url,
                            root_url=root_link,
                            visited=visited,
                        )
                    )

                processed_links = await asyncio.gather(*tasks_process_link)
                extracted_urls = await asyncio.gather(*tasks_extract_urls)

                docs.extend([link for link in processed_links if link[0] is not None])
                queue.extend([url for extracted_urls_from_one in extracted_urls for url in extracted_urls_from_one])
                depth += 1

        logger.info(f"Found {len(docs)} documents on {root_link}")
        return docs[:max_total_docs]

    async def traverse_xml(self, link: str) -> List[Tuple[Doc, DocumentMetadata, str]]:
        queue, visited, depth = deque([link]), set([link]), 0
        docs = []

        async with ClientSession() as session:
            while queue:
                logger.info(f"Depth: {depth}, total: {len(docs)}, queue size: {len(queue)}, link: {link}")
                tasks_process_link = []
                tasks_extract_urls = []
                for _ in range(len(queue)):
                    url = queue.popleft()
                    if re.search(self.sitemap_pattern, url):
                        logger.info(f"Extracting links from .xml '{url}'")
                        tasks_extract_urls.append(
                            self.extract_urls(
                                session=session,
                                link=url,
                                visited=visited,
                            )
                        )
                    else:
                        tasks_process_link.append(self.process_link(session=session, link=url))

                processed_links = await asyncio.gather(*tasks_process_link)
                extracted_urls = await asyncio.gather(*tasks_extract_urls)

                docs.extend([(doc, medatadata, content) for (doc, medatadata, content) in processed_links if doc])
                queue.extend([url for extracted_urls_from_one in extracted_urls for url in extracted_urls_from_one])
                depth += 1

        logger.info(f"Found {len(docs)} documents on {link}")
        return docs

    async def link_to_docs(
        self, link: str, vendor: str, organization: str, collection: str, ignore_urls: bool = True
    ) -> Tuple[List[Doc], List[DocumentMetadata]]:
        default_ignore_links = self.converter.ignore_links
        self.converter.ignore_links = ignore_urls

        # If we face sitemap, we will use links from it for extraction.
        # Otherwise, we are recursively crawling page.
        if re.search(self.sitemap_pattern, link):
            results = await self.traverse_xml(link)
        else:
            results = await self.traverse_page(link)

        documents, documents_metadata = [], []
        for doc, metadata, content in results:
            documents.append(doc)
            documents_metadata.append(metadata)

            # Writing plain page content to GridFS
            filename = full_collection_name(vendor, organization, collection) + "_" + metadata.id
            res = GRIDFS.find_one({"filename": filename})
            if res:
                GRIDFS.delete(res._id)
                logger.info(f"Deleted file {filename} from GridFS")
            GRIDFS.put(
                content.encode(),
                filename=filename,
                content_type="text/html",
            )

        self.converter.ignore_links = default_ignore_links

        return documents, documents_metadata

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

        filename = full_collection_name(vendor, organization, collection) + "_" + doc_id
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
                translation = AWS_TRANSLATE_CLIENT.translate_text(text=document.content)
                meta["source_language"] = translation["source_language"]
                content = translation["translation"]
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
                translation = AWS_TRANSLATE_CLIENT.translate_text(raw_lines)
                text_lines = [f"{ent[0].role}: {ent[1]}" for ent in zip(document.history, translation["translation"])]
                meta["source_language"] = translation["source_language"]
            content = "\n".join(text_lines)
            chunks = self.chat_to_chunks(text_lines)
        return chunks, meta, content
