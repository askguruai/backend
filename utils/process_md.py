import os

from parsers.markdown_parser import MarkdownParser
from utils.ml_requests import get_embeddings
from argparse import ArgumentParser


def process_single_file(path, collection, parser: MarkdownParser):

    chunks = parser.process_file(path)
    embeddings = get_embeddings(chunks, api_version="v1")
    assert len(embeddings) == len(chunks)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--source_dir", type=str)
    args = parser.parse_args()

    md_parser = MarkdownParser(1024)
    docs = os.listdir(args.source_dir)[:2]
    for doc in docs:
        process_single_file(os.path.join(args.source_dir, doc), None, md_parser)


