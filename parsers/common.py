from nltk import tokenize
from typing import List


def text_to_sentences(text: str) -> List[str]:
    return tokenize.sent_tokenize(text)


def chunkise_sentences(sentences: List[str], chunk_size: int = 1024) -> List[str]:
    chunks = []
    current_cunk = ""
    for sent in sentences:
        if len(current_cunk) + len(sent) <= chunk_size:
            current_cunk += sent
        else:
            chunks.append(current_cunk)
            current_cunk = sent
    chunks.append(current_cunk)
    return chunks