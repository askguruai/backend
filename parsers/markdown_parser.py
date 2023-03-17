from copy import deepcopy
from typing import Any, Union, List, Tuple
import marko
import re

from parsers.general_parser import GeneralParser
from marko.block import Paragraph, Heading, Document, BlankLine, ThematicBreak, List as MDList, ListItem, \
    FencedCode, HTMLBlock, Quote, CodeBlock
from marko.inline import RawText, Emphasis, StrongEmphasis, Link, Literal, LineBreak


class MarkdownParser(GeneralParser):
    def get_text(self, path: str) -> str:
        with open(path, "rt") as f:
            text = f.read()
        return text

    def process_file(self, path) -> Tuple[List[str], str]:
        text = self.get_text(path)
        text, meta = self.preprocess_text(text)
        document = marko.parse(text)
        document = self.preprocess_document(document)

        chunks = []
        accumulated_text = ""
        current_heading = ""

        for child in document.children:
            if isinstance(child, Heading):
                heading_text = self.render_heading(child)
                if child.level == 1 or child.level == 2:
                    # creating text chunk from level-1/2 section
                    if len(accumulated_text) > 0:
                        chunk_text = f"{current_heading}\n{accumulated_text}" if current_heading != "" else accumulated_text
                        accumulated_text = ""
                        chunks.append({
                            "title": meta,
                            "text": chunk_text
                        })
                    current_heading = heading_text
                else:
                    accumulated_text += f"\n{self.render_text(child)}\n"
            else:
                cur_text = self.render_text(child)
                if len(accumulated_text) + len(cur_text) > 2000:
                    if accumulated_text != "":
                        chunks.append({
                            "title": meta,
                            "text": f"{current_heading}\n{accumulated_text}"
                        })
                    accumulated_text = cur_text
                else:
                    accumulated_text += cur_text

        if len(accumulated_text) > 0:
            chunk_text = f"{current_heading}\n{accumulated_text}" if current_heading != "" else accumulated_text
            chunks.append({
                "title": meta,
                "text": chunk_text
            })

        chunks = self.compress_chunks(chunks)
        return ["\n".join([ch["title"], ch["text"]]) for ch in chunks], meta

    def preprocess_text(self, text: str):
        # removing placeholder tags
        text = re.sub(r"{{% *ol *%}}", "1. ", text)
        for i in range(1, 20):
            text = re.sub(f'{{{{% *ol *start="{i}" *%}}}}', f'{i}. ', text)
        text = re.sub(r"{{%.*?%}}", "", text)
        text = re.sub(r"{#.*?}", "", text)
        text = re.sub(r"{{<.*?>}}", "", text)

        elem_meta_re = re.compile(r"---(.*)?---", re.DOTALL)
        search = re.search(elem_meta_re, text)
        meta = ""
        if search:
            matched = search.group(1).strip().split('\n')
            for line in matched:
                key, val = line.split(":", maxsplit=1)
                if key.strip() == "title":
                    meta = val.strip()
                    break
        elem_meta_re = re.compile(r"---.*---", re.DOTALL)
        text = re.sub(elem_meta_re, "", text)
        return text, meta

    def preprocess_document(self, document: Document):
        new_children = [ch for ch in document.children if not isinstance(ch, (ThematicBreak, HTMLBlock))]
        document.children = new_children
        return document

    def extract_text(self, elem):
        text_parts = []
        for ch in elem.children:
            if isinstance(ch, (RawText, Literal)):
                text_parts.append(ch.children)
            elif isinstance(ch, LineBreak):
                text_parts.append("\n")
            elif isinstance(ch, (Emphasis, StrongEmphasis, Link)):
                text_parts.extend(self.extract_text(ch))
        text = "".join(text_parts)
        return text.strip()

    def render_quote(self, blockquote: Quote):
        quote_parts = []
        for ch in blockquote.children:
            rendered = self.render_text(ch)
            quote_parts.append(rendered)
        return "".join(quote_parts)

    def render_code(self, codeblock: Union[FencedCode, CodeBlock]):
        return f"\n{self.extract_text(codeblock)}\n"

    def render_paragraph(self, para: Paragraph):
        return self.extract_text(para)

    def render_heading(self, heading: Heading):
        return f"{self.extract_text(heading)}"

    def render_list(self, list: MDList):
        items = []
        list_start = list.start
        for i, list_item in enumerate(list.children, start=list_start):
            assert isinstance(list_item, ListItem)
            item_text = ""
            for child in list_item.children:
                item_text += self.render_text(child)
            pref = f"{i}. " if list.ordered else "* "
            items.append(f"{pref}{item_text}\n")
        return "".join(items)

    def render_text(self, element):
        if isinstance(element, Paragraph):
            rendered = self.render_paragraph(element)
        elif isinstance(element, Heading):
            rendered = self.render_heading(element)
        elif isinstance(element, MDList):
            rendered = self.render_list(element)
        elif isinstance(element, (FencedCode, CodeBlock)):
            rendered = self.render_code(element)
        elif isinstance(element, BlankLine):
            rendered = "\n"
        elif isinstance(element, Quote):
            rendered = self.render_quote(element)
        else:
            # fallback
            print(f"Unknown element: {element.__class__}")
            rendered = marko.render(element)
        return rendered

    def compress_chunks(self, chunks: list):
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
