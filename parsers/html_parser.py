import re
from typing import Tuple, List

import requests
from argparse import ArgumentParser
import json
import htmltabletomd
from copy import deepcopy
from collections import deque
import abc

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


class TagPartsList(list):
    def __init__(self):
        super().__init__()

    def app_(self, part: List[str]):
        if len(self) == 0:
            self.append(part)
        else:
            if part[1] == "header":
                self.append(part)
            else:
                self[-1][0] += part[0]


class ChunksManager:
    def __init__(self, meta: str, chunk_max_length: int = 2000):
        self.meta = meta
        self.chunk_max_length = chunk_max_length
        self.current_header = ""
        self.accumulated_text = ""
        self.chunks = []

    def update(self, newparts: List[List[str]]):
        for part in newparts:
            cur_text, type_ = part
            if type_ == "header":
                if self.accumulated_text != "":
                    chunk_text = (
                        f"{self.current_header}\n{self.accumulated_text}"
                        if self.current_header != ""
                        else self.accumulated_text
                    )
                    self.accumulated_text = ""
                    chunk = {"title": self.meta, "text": chunk_text}
                    self.chunks.extend(self.opt_split_into_smaller_chunks(chunk))
                self.current_header = cur_text
            elif type_ == "text":
                if len(self.accumulated_text) + len(cur_text) > self.chunk_max_length:
                    if self.accumulated_text != "":
                        chunk_text = (
                            f"{self.current_header}\n{self.accumulated_text}"
                            if self.current_header != ""
                            else self.accumulated_text
                        )
                        chunk = {"title": self.meta, "text": chunk_text}
                        self.chunks.extend(self.opt_split_into_smaller_chunks(chunk))
                    self.accumulated_text = cur_text
                else:
                    self.accumulated_text += f"\n{cur_text}"
            else:
                assert False, f"Unknown type {type_}"

    def compress_chunks(self, chunks: list) -> list:
        new_chunks = []
        cur_chunk = chunks[0]
        for chunk in chunks[1:]:
            if len(cur_chunk["text"]) + len(chunk["text"]) < self.chunk_max_length:
                cur_chunk["text"] += f"\n{chunk['text']}"
            else:
                new_chunks.append(deepcopy(cur_chunk))
                cur_chunk = chunk
        new_chunks.append(cur_chunk)
        return new_chunks

    def opt_split_into_smaller_chunks(self, chunk: dict) -> list[dict]:
        # todo: split by \n
        smaller_chunks = []
        while len(chunk["text"]) > 3000:
            part = {"title": chunk["title"],
                    "text": chunk["text"][:3000]}
            smaller_chunks.append(part)
            chunk["text"] = chunk["text"][3000:]
        smaller_chunks.append(chunk)
        if len(smaller_chunks) > 1:
            print(f"Hard split happened! Number of chunks: {len(smaller_chunks)}")
        return smaller_chunks

    def digest(self):
        if self.accumulated_text != "":
            chunk_text = (
                f"{self.current_header}\n{self.accumulated_text}"
                if self.current_header != ""
                else self.accumulated_text
            )
            chunk = {"title": self.meta, "text": chunk_text}
            self.chunks.extend(self.opt_split_into_smaller_chunks(chunk))
        if len(self.chunks) == 0:
            return []
        chunks = self.compress_chunks(self.chunks)
        return chunks


class HTMLParser:
    def __init__(self, chunk_length = 1024):
        self.chunk_length = chunk_length

    def render_list(self, elem: Tag, depth=0):
        # print(f"Rendering list {elem.name} with depth {depth}")
        ordered = elem.name == "ol"
        items = []
        cnter = 1
        for ch in elem.children:
            if isinstance(ch, NavigableString):
                continue
            if ch.name in ["ol", "ul"]:
                items.append(self.render_list(ch, depth=depth+1))
                continue
            elif ch.name == "br":
                continue
            elif ch.name in ["p", "h1", "h2", "h3", "h4", "h5", "h6", "div"]:
                # bruh what the fuck
                items.append(self.render_text(ch))
                continue
            elif ch.name in ["img"]:
                continue
            assert ch.name in ["li", "option"], (ch.name, type(ch))
            # print("rendering li")
            pref = " " * (2*depth) + (f"{cnter}. " if ordered else "* ")
            # print(f"_______Pref: {pref}")
            text_within = self.render_text(ch, list_depth=depth+1)
            # print(text_within.__repr__())
            # exit(0)
            items.append(f"{pref}{text_within}")
            cnter += 1
        return "\n".join(items)

    def render_table(self, table: str) -> str:
        return htmltabletomd.convert_table(table)

    def render_link(self, elem: Tag) -> str:
        if "href" not in elem.attrs:
            return self.render_text(elem)
        link = elem.attrs['href']
        if link.startswith("#"):
            return self.render_text(elem)
        else:
            return f"{self.render_text(elem)} (link:{elem.attrs['href']})"

    def render_code(self, elem: Tag) -> str:
        return f"\n{self.render_text(elem)}\n"

    def render_text(self, elem: Tag, list_depth=0):
        parts = []
        for ch in elem.children:
            if ch.name is None:
                rendered = ch
            elif ch.name in ["ul", "ol"]:
                rendered = self.render_list(ch, depth=list_depth)
            elif ch.name == "pre":
                rendered = "".join(ch.strings) + "\n"
            elif ch.name == "a":
                rendered = self.render_link(ch)
            elif ch.name in ["span", "code", "em", "strong", "p", "small", "div", "sub", "sup",
                             "label", "time", "u", "b", "span2.", "i", "center", "var", "font", "option"]:
                rendered = self.render_text(ch)
            elif ch.name.startswith("st1:") or \
                    ch.name.startswith("mailto") or \
                    ch.name.startswith("http"):  # vivantio specificity
                rendered = self.render_text(ch)
            elif ch.name in ["br", "hr"]:
                rendered = "\n"
            elif ch.name in ["img", "iframe", "style"]:
                continue
            elif ch.name == "table":
                rendered = ""
                # rendered = self.render_table(str(ch))
            elif ch.name in ["script"]:
                rendered = self.render_code(ch)
            elif ch.name in ["h1", "h2"]:
                rendered = f"%%%%%{self.render_text(ch)}%%%%%"
            elif ch.name in ["h3", "h4", "h5", "h6"]:
                # ffs what kind of psycho puts headers in paragraphs
                rendered = f"\n{self.render_text(ch)}"
            elif ch.name in ["li"]:
                # print(f"li outinside of the list!!!")
                rendered = f"\n* {self.render_text(ch)}"
            elif ch.name in ["o:p", 'br=""', "source", "audio", "video", "input", "picture", "w:wrap", "o:o:p",
                             "noscript", "table"]:
                continue
            elif ch.name.startswith("v:"):  # vivantio specificity
                continue
            else:
                assert False, f"Unknown tag parsing text: {ch.name}: {self.render_text(ch)}"
            # print(f"Rendering {ch.name}, adding {rendered.__repr__()}")
            parts.append(rendered)
        return "".join(parts).strip()

    def render_tag(self, elem: Tag) -> str:
        if elem.name in ["ul", "ol", "select"]:
            list_text = self.render_list(elem)
            return list_text
        elif elem.name == "a":
            link_text = self.render_link(elem)
            return link_text
        elif elem.name in ["p"]:
            p_text = self.render_text(elem)
            return p_text
        elif elem.name in ["div", "blockquote", "strong", "br", "hr", "span", "b", "font", "u", "i",
                           "center", "em", "option", "sup"]:
            elem_text = self.render_text(elem)
            return elem_text
        elif elem.name == "pre":
            pre_text = "".join(elem.strings)
            return pre_text
        elif elem.name == "table":
            # temporary because of vivantio
            table_text = ""
            # table_text =  self.render_table(str(elem))
            return table_text
        elif elem.name == "li":
            li_text = f"\n*{self.render_text(elem)}"
            return li_text
        elif elem.name in ["style", "meta", "noscript", "title", "link", "style", "form", "button", "figure",
                           "header", "video", "script", "database"]:
            return ""
        elif elem.name in ["code"]:
            code_text = self.render_code(elem)
            return code_text
        elif elem.name.startswith("mailto"):  # vivantio
            text = self.render_text(elem)
            return text
        elif elem.name.startswith("w:") or\
                elem.name.startswith("http"):  # vivantio
            return ""
        else:
            # print(f"Unknown element with name {elem.name}!: {render_text(elem)}")
            print(f"Unknown tag with name {elem.name}!: {''.join(elem.strings)}")
            return "".join(elem.strings)

    @abc.abstractmethod
    def preprocess_document(self, article: dict) -> Tuple[str, str, dict]:
        pass

    def _extract_headers(self, text: str):
        # todo: more elegant idk
        regex = re.compile(r"%%%%%(.+?)%%%%%")
        start_poss, end_poss = [], []
        for m in regex.finditer(text):
            start_poss.append(m.start())
            end_poss.append(m.end())
        if len(start_poss) == 0:
            return [[text, "text"]]
        text_start_poss = [0]
        text_end_poss = []
        for s, e in zip(start_poss, end_poss):
            text_end_poss.append(s)
            text_start_poss.append(e)
        text_end_poss.append(len(text))
        assert len(text_start_poss) == len(text_end_poss)
        parts = []

        for i in range(len(start_poss)):
            text_part = text[text_start_poss[i]:text_end_poss[i]]
            if len(text_part) > 0:
                parts.append([text_part, "text"])
            header_part = text[start_poss[i]: end_poss[i]]
            header_part = header_part[5:-5]
            parts.append([header_part, "header"])
        text_part = text[text_start_poss[-1]: text_end_poss[-1]]
        if len(text_part) > 0:
            parts.append([text_part, "text"])
        return parts


    def process_document(self, article: dict, debug=False):
        meta, body, meta_info = self.preprocess_document(article)

        parsed = BeautifulSoup(body)
        if debug:
            print(parsed)
        # print("".join(parsed.strings))
        queue = deque(parsed.children)
        chunks_manager = ChunksManager(meta)
        while len(queue) > 0:
            ch = queue.popleft()
            if isinstance(ch, Tag):
                if ch.name == "div":
                    queue.extendleft(list(ch.children)[::-1])
                    continue
                elif ch.name in ["h1", "h2"]:
                    if debug:
                        print(f"\n______________________\nrendering tag: {ch.name}")
                    heading_text = self.render_text(ch)
                    chunks_manager.update([[heading_text, "header"]])
                    if debug:
                        print(f"\nheader processed: {heading_text}\n___________________")
                elif ch.name in ["h3", "h4", "h5", "h6"]:
                    heading_text = f"\n\n{self.render_text(ch)}"
                    chunks_manager.update([[heading_text, "text"]])
                elif ch.name in ["iframe", "img"]:
                    continue
                else:
                    if debug:
                        print(f"\n______________________\nrendering tag: {ch.name}")
                    tag_text = self.render_tag(ch)
                    parts = self._extract_headers(tag_text)
                    chunks_manager.update(parts)

                    # chunks_manager.update(tag_parts)
                    # if debug:
                    #     print(f"\ntag parts: {tag_parts}\n___________________")

        chunks = chunks_manager.digest()

        if len(chunks) == 0:
            print(f"Warning: article with zero chunks! {meta}")
            return [], None
        return ["\n".join([ch["title"], ch["text"]]).strip() for ch in chunks], meta_info


class GrooveHTMLParser(HTMLParser):
    def __init__(self, chunk_length=1024):
        super().__init__(chunk_length)

    def preprocess_document(self, article: dict) -> Tuple[str, str, dict]:
        meta = [article["title"]]
        if len(article['tags']) > 0:
            meta.append(f"tags: {','.join(article['tags'])}")
        if len(article['related_titles']) > 0:
            meta.append(f"related: {','.join(article['related_titles'])}")
        meta = "\n".join(meta)

        meta_info = {
            "title": article["title"],
            "slug": article["slug"],
            "id": article["id"]
        }

        body = article["body"]
        return meta, body, meta_info


class VivantioHTMLParser(HTMLParser):
    def __init__(self, chunk_length=200, companyname="vivantio"):
        super().__init__(chunk_length)
        self.companyname = companyname

    def preprocess_document(self, article: dict) -> Tuple[str, str, dict]:
        meta = [article["Title"]]

        if "CategoryName" in article and article["CategoryName"] is not None:
            meta.append(article["CategoryName"])
        if "ThemeName" in article and article["ThemeName"] is not None:
            meta.append(article["ThemeName"])
        if len(article["Keywords"]) > 0:
            meta.append(f"tags: {','.join(article['Keywords'])}")
        meta = "\n".join(meta)

        meta_info = {
            "title": article["Title"],
            "id": article["Id"],
            "link": f"https://{self.companyname}.flex.vivantio.com/item/Article/{article['Id']}"
        }

        body = article["Text"]
        return meta, body, meta_info
