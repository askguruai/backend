from argparse import ArgumentParser
import fitz
from pathlib import Path
from nltk import tokenize
from typing import List, Union
from parsers.common import text_to_sentences, chunkise_sentences


def parse_document(path: Union[str, Path], chunk_size: int = 1024) -> List[str]:
    # TODO: split big pieces of text into smaller ones?

    with fitz.open(path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    sentences = text_to_sentences(text)
    sentences = [sent.replace("\n", " ") for sent in sentences]
    return chunkise_sentences(sentences)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", help=f"Path to a .pdf file")
    args = parser.parse_args()
    chunks = parse_document(args.file)
    for chunk in chunks:
        print(f"{chunk} --- {len(chunk)}")
        print()
