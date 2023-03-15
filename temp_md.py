import re
import sys

import marko
from marko.block import Paragraph, Heading, Document, BlankLine, ThematicBreak, List as MDList, ListItem
from marko.inline import RawText, Emphasis, StrongEmphasis
from copy import deepcopy


def preprocess_text(text: str):
    # removing placeholder tags
    text = re.sub(r"{{% *ol *%}}", "1. ", text)
    for i in range(1, 20):
        text = re.sub(f'{{{{% *ol *start="{i}" *%}}}}', f'{i}. ', text)
    text = re.sub(r"{{%.*?%}}", "", text)

    # compressing metainfo section
    meta = []
    for key, regex in [
        ("title", "title: (.*)date"),
        # ("description", "description:(.*)---")
    ]:
        reg = re.compile(regex, re.DOTALL)
        search = re.search(reg, text)
        if search:
            matched = search.group(1).strip().replace('\n', '')
            meta.append(f"{key}: {matched}")
    elem_meta_re = re.compile(r"---.*---", re.DOTALL)
    text = re.sub(elem_meta_re, "", text)
    meta = "\n".join(meta)
    return text, meta


def extract_text(elem):
    text_parts = []
    for ch in elem.children:
        if isinstance(ch, RawText):
            text_parts.append(ch.children)
        elif isinstance(ch, (Emphasis, StrongEmphasis)):
            text_parts.extend(extract_text(ch))
    text = "".join(text_parts)
    return text.strip()


def preprocess_document(document: Document):
    new_children = [ch for ch in document.children if not isinstance(ch, (BlankLine, ThematicBreak))]
    document.children = new_children
    return document


def render_text(element):
    if isinstance(element, Paragraph):
        return render_paragraph(element)
    elif isinstance(element, Heading):
        return render_heading(element)
    elif isinstance(element, MDList):
        return render_list(element)
    else:
        # fallback
        print(f"Unknown element: {element.__class__}")
        return marko.render(element)


def render_paragraph(para: Paragraph):
    return extract_text(para)


def render_heading(heading: Heading):
    return f"{extract_text(heading)}"


def render_list(list: MDList):
    items = []
    list_start = list.start
    for i, list_item in enumerate(list.children, start=list_start):
        assert isinstance(list_item, ListItem)
        item_text = ""
        for child in list_item.children:
            item_text += render_text(child)
        pref = f"\n{i}. " if list.ordered else "\n* "
        items.append(f"{pref}{item_text}")
    return "".join(items)


def compress_chunks(chunks: list):
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


with open(sys.argv[1], "rt") as f:
    text = f.read()
text, meta = preprocess_text(text)
document = marko.parse(text)
document = preprocess_document(document)

chunks = []
accumulated_text = []
current_heading = ""
for child in document.children:
    if isinstance(child, Heading):
        heading_text = render_heading(child)
        if child.level == 1 or child.level == 2:
            # creating text chunk from level-1/2 section
            if len(accumulated_text) > 0:
                chunk_text = ""
                if current_heading != "":
                    chunk_text += f"{current_heading}\n"
                chunk_text += "".join(accumulated_text)
                accumulated_text = []
                chunks.append({
                    "title": meta,
                    "text": chunk_text
                })
            current_heading = heading_text
        else:
            accumulated_text.append(f"\n{render_text(child)}\n")
    else:
        accumulated_text.append(render_text(child))

if len(accumulated_text) > 0:
    chunk_text = ""
    if current_heading != "":
        chunk_text += f"{current_heading}\n"
    chunk_text += "".join(accumulated_text)
    accumulated_text = []
    chunks.append({
        "title": meta,
        "text": chunk_text
    })

chunks = compress_chunks(chunks)
chunks = ["\n".join([ch["title"], ch["text"]]) for ch in chunks]

for chunk in chunks:
    print(chunk)
    print(f"___________{len(chunk)}___________\n")
