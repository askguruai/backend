from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import fitz
from nltk import tokenize

from parsers.common import chunkise_sentences, text_to_sentences


def parse_document(path: Union[str, Path], chunk_size: int) -> List[str]:
    # TODO: split big pieces of text into smaller ones?

    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    sentences = text_to_sentences(text)
    sentences = [sent.replace("\n", " ") for sent in sentences]
    return chunkise_sentences(sentences, chunk_size)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help=f"Path to a .pdf file")
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()
    chunks = parse_document(args.file, args.chunk_size)
    for chunk in chunks:
        print(f"{chunk} --- {len(chunk)}")
        print()
