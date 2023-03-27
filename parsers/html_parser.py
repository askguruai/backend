from typing import Tuple

import requests
from argparse import ArgumentParser
import json
import htmltabletomd
from copy import deepcopy
from collections import deque

from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString


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
            assert ch.name == "li", (ch.name, type(ch))
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
                rendered = f"{self.render_text(ch)} (link:{ch.attrs['href']})"
            elif ch.name in ["span", "code", "em", "strong", "p", "small", "div"]:
                rendered = self.render_text(ch)
            elif ch.name in ["br", "hr"]:
                rendered = "\n"
            elif ch.name in ["img", "iframe"]:
                continue
            elif ch.name == "table":
                rendered = self.render_table(str(ch))
            else:
                assert False, f"Unknown tag parsing text: {ch.name}: {self.render_text(ch)}"
            # print(f"Rendering {ch.name}, adding {rendered.__repr__()}")
            parts.append(rendered)
        return "".join(parts).strip()

    def render_tag(self, elem: Tag) -> str:
        if elem.name == "ul" or elem.name == "ol":
            return self.render_list(elem)
        elif elem.name == "a":
            return f"{self.render_text(elem)} (link:{elem.attrs['href']})"
        elif elem.name in ["p", "div", "blockquote", "strong", "br", "hr"]:
            return self.render_text(elem)
        elif elem.name == "pre":
            return "".join(elem.strings)
        elif elem.name == "table":
            return self.render_table(str(elem))
        else:
            # print(f"Unknown element with name {elem.name}!: {render_text(elem)}")
            print(f"Unknown element with name {elem.name}!: {''.join(elem.strings)}")
            # rendering default
            return "".join(elem.strings)


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


    def compress_chunks(self, chunks: list) -> list:
        new_chunks = []
        cur_chunk = chunks[0]
        for chunk in chunks[1:]:
            if len(cur_chunk["text"]) + len(chunk["text"]) < 1024:
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
            print(f"Hard split happened!")
        return smaller_chunks

    def process_document(self, article: dict, debug=False):
        meta, body, meta_info = self.preprocess_document(article)

        parsed = BeautifulSoup(body)
        if debug:
            print(parsed)
        # print("".join(parsed.strings))
        chunks = []
        accumulated_text = ""
        current_heading = ""
        queue = deque(parsed.children)
        while len(queue) > 0:
            ch = queue.popleft()
            if isinstance(ch, Tag):
                if ch.name == "div":
                    queue.extendleft(list(ch.children)[::-1])
                    continue
                elif ch.name in ["h1", "h2"]:
                    heading_text = self.render_text(ch)
                    if len(accumulated_text) > 0:
                        chunk_text = (
                            f"{current_heading}\n{accumulated_text}"
                            if current_heading != ""
                            else accumulated_text
                        )
                        accumulated_text = ""
                        chunk = {"title": meta, "text": chunk_text}
                        chunks.extend(self.opt_split_into_smaller_chunks(chunk))
                    current_heading = heading_text
                elif ch.name in ["h3", "h4", "h5", "h6"]:
                    accumulated_text += f"\n\n{self.render_text(ch)}"
                elif ch.name in ["iframe", "img"]:
                    continue
                else:
                    if debug:
                        print(f"rendering tag: {ch.name}")
                    cur_text = self.render_tag(ch)
                    if len(accumulated_text) + len(cur_text) > 2000:
                        if accumulated_text != "":
                            chunk = {"title": meta, "text": f"{current_heading}\n{accumulated_text}"}
                            chunks.extend(self.opt_split_into_smaller_chunks(chunk))
                        accumulated_text = cur_text
                    else:
                        accumulated_text += f"\n{cur_text}"
        if len(accumulated_text) > 0:
            chunk_text = (
                f"{current_heading}\n{accumulated_text}"
                if current_heading != ""
                else accumulated_text
            )
            chunk = {"title": meta, "text": chunk_text}
            chunks.extend(self.opt_split_into_smaller_chunks(chunk))

        # print(f"Chunks before compression: {len(chunks)}")
        chunks = self.compress_chunks(chunks)
        # print(f"Chunks after compression: {len(chunks)}")
        return ["\n".join([ch["title"], ch["text"]]) for ch in chunks], meta_info

