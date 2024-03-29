import os.path as osp
from tempfile import TemporaryDirectory
from typing import List, Tuple

import docx
from simplify_docx import simplify

from parsers.general_parser import GeneralParser
from utils.tokenize_ import doc_to_chunks

# from docx_parser import DocumentParser


class ListingDispatcher:
    def __init__(self, numbered: bool = False) -> None:
        self.numbered = numbered
        self.nums = {}

    def reset(self):
        self.nums = {}

    def get_prefix(self, ilevel: int):
        if ilevel in self.nums:
            num = self.nums[ilevel]
            self.nums[ilevel] += 1
        else:
            num = 1
            self.nums[ilevel] = 2
        # invalidate higher nums
        for i in range(ilevel + 1, 8):
            if i in self.nums:
                del self.nums[i]
        indent = "  " * ilevel
        pref = f"{num}. " if self.numbered else "* "
        return f"{indent}{pref}"


class DocxParser(GeneralParser):
    def __init__(self, chunk_size: int):
        super().__init__(chunk_size)
        self.ld = None

    def process_file(self, path, content_only=False) -> Tuple[List[str], str]:
        file_name = osp.splitext(osp.split(path)[1])[0]
        meta = {"doc_title": file_name}
        contents = []
        doc = docx.Document(path)
        structure = simplify(doc)
        body = None
        for ent in structure["VALUE"]:
            if ent["TYPE"] == "body":
                body = ent["VALUE"]
        assert body is not None, f"Body not found parsing {path}"
        for p in body:
            if p["TYPE"] == "paragraph":
                paragraph_text = self.parse_paragraph(p)
                contents.append(paragraph_text)
        content = "\n".join(contents)
        if content_only:
            return content
        meta["security_groups"] = 2**63 - 1
        chunks = doc_to_chunks(content=content, title=file_name)
        return chunks, content, meta

    def parse_paragraph(self, paragraph: dict):
        text = ""
        for item in paragraph["VALUE"]:
            if item["TYPE"] == "text":
                text += item["VALUE"]
        if "style" in paragraph:
            if "numPr" in paragraph["style"]:
                # we faced a list item
                if self.ld is None:
                    self.ld = ListingDispatcher(numbered=(paragraph["style"]["numPr"]["numId"] in [0, 2]))
                prefix = self.ld.get_prefix(paragraph["style"]["numPr"]["ilvl"])
                text = f"{prefix}{text}"
            else:
                self.ld = None
        else:
            # resetting dispatcher
            self.ld = None
        return text

    def stream2text(self, stream: bytes) -> str:
        with TemporaryDirectory() as tmp:
            with open(osp.join(tmp, "document.docx"), "wb") as f:
                f.write(stream)
            return self.process_file(osp.join(tmp, "document.docx"), content_only=True)


# my_doc = docx.Document(infile)
# my_doc_as_json = simplify(my_doc)
# with open('temp_doc2json.json','wt') as f:
#     json.dump(my_doc_as_json, f, indent=4)
