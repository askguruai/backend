from typing import List, Tuple

from utils.misc import int_list_encode
from utils.schemas import Chat


class ChatParser:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def process_document(self, chat: Chat) -> Tuple[List[str], dict]:
        meta = {
            "doc_id": chat.id,
            "doc_title": f"{chat.user.name}::{chat.user.id}",
            "timestamp": int(chat.timestamp),
            "security_groups": int_list_encode(chat.security_groups),
        }
        text = [f"{message.role}: {message.content}" for message in chat.history]
        return self.to_chunks(text), meta

    def to_chunks(self, text_lines: List[str]) -> List[str]:
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
