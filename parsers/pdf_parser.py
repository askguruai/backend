from argparse import ArgumentParser
from pathlib import Path
from typing import List, Union

import fitz

from parsers.common import chunkise_sentences, text_to_sentences


def extract_content(path: Union[str, Path]):
    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def parse_pdf_content(content: str, chunk_size: int) -> List[str]:
    sentences = text_to_sentences(content)
    sentences = [sent.replace("\n", " ") for sent in sentences]
    return chunkise_sentences(sentences, chunk_size)


def parse_document(path: Union[str, Path], chunk_size: int) -> List[str]:
    return parse_pdf_content(extract_content(path), chunk_size)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help=f"Path to a .pdf file")
    parser.add_argument("--chunk_size", type=int, default=512)
    args = parser.parse_args()
    chunks = parse_document(args.file, args.chunk_size)
    for chunk in chunks:
        print(f"{chunk} --- {len(chunk)}")
        print()
