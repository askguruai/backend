from datetime import datetime
from typing import List, Tuple

import tiktoken

from utils import hash_string
from utils.misc import int_list_encode
from utils.schemas import Chat, Doc


class DocumentsParser:
    def __init__(self, chunk_size: int, tokenizer_name: str):
        self.chunk_size = chunk_size
        self.enc = tiktoken.get_encoding(tokenizer_name)

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
            chunks = self.doc_to_chunks(document.content)
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

    def doc_to_chunks(self, text: str) -> List[str]:
        chunks = []

        current_content, part = "", 0
        # TODO: split by lines which are bolded
        # because they are the titles of the sections
        for line in content.split("\n"):
            if len(self.enc.encode(current_content + line + "\n")) > self.chunk_size:
                chunks.append(current_content.strip())
                current_content = ""
                part += 1
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
