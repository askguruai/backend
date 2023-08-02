import os.path as osp
import re
from typing import List, Tuple

import fitz

from parsers.general_parser import GeneralParser
from utils.tokenize_ import doc_to_chunks


class PdfParser(GeneralParser):
    def __init__(self, chunk_size: int):
        super().__init__(chunk_size)
        self.ld = None

    def process_file(self, path) -> Tuple[List[str], str, dict]:
        file_name = osp.splitext(osp.split(path)[1])[0]
        meta = {"doc_title": file_name}
        with fitz.open(path) as doc:
            content = ""
            for page in doc:
                content += page.get_text()
        content = re.sub(r"\.\.\.\.+", "...", content)
        meta["security_groups"] = 2**63 - 1
        chunks = doc_to_chunks(content=content, title=file_name)
        return chunks, content, meta

    def stream2text(self, stream: bytes) -> str:
        with fitz.open(stream=stream, filetype="pdf") as fl:
            text = ""
            for page in fl:
                text += page.get_text()
        return text
