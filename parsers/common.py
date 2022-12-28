import logging
from nltk import tokenize
from typing import List


def text_to_sentences(text: str) -> List[str]:
    return tokenize.sent_tokenize(text)


def chunkise_sentences(sentences: List[str], chunk_size: int) -> List[str]:
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            chunks.append(current_chunk)
            current_chunk = sent
    chunks.append(current_chunk)
    return chunks
