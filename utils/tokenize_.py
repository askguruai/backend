import time
from collections import deque
from typing import List

import nltk
import tiktoken

from utils import CONFIG

TOKEIZERS = {}


def get_tokenizer(tokenizer_name: str):
    if tokenizer_name not in TOKEIZERS:
        TOKEIZERS[tokenizer_name] = tiktoken.get_encoding(tokenizer_name)
    return TOKEIZERS[tokenizer_name]


def doc_to_chunks(
    content: str,
    title: str = "",
    summary: str = "",
    tokenizer_name: str = CONFIG["handlers"]["tokenizer_name"],
    chunk_size: int = int(CONFIG["handlers"]["chunk_size"]),
    overlapping_lines: int = 5,
) -> List[str]:
    chunks = []
    encoder = get_tokenizer(tokenizer_name)
    olap = deque([], overlapping_lines)

    current_content, part = "", 0
    # TODO: split by lines which are bolded (in case of cars)
    # because they are the titles of the sections
    for line in content.split("\n"):
        if len(encoder.encode(current_content + line + "\n")) > chunk_size:
            chunks.append(current_content.strip())
            current_content = f"{' '.join(olap)}\n"
            part += 1

        if len(encoder.encode(line)) > chunk_size:
            # splitting paragraph into sentences
            sentences = nltk.tokenize.sent_tokenize(line)
            for sent in sentences:
                if len(encoder.encode(current_content + f" {sent}")) > chunk_size:
                    chunks.append(current_content.strip())
                    current_content = f"{'. '.join(olap)}\n{sent}"
                else:
                    current_content += f" {sent}"
                olap.extend(sent)

            # should reapply the paragraph split
            current_content += "\n"

        else:
            current_content += line + "\n"
            # now need to append a few last sentences to our deque
            sentences = nltk.tokenize.sent_tokenize(line)
            olap.extend(sentences[-overlapping_lines:])

    chunks.append(current_content.strip())
    return chunks


# preload config tokenizer
TOKEIZERS[CONFIG["handlers"]["tokenizer_name"]] = get_tokenizer(CONFIG["handlers"]["tokenizer_name"])
