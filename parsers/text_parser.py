from parsers.common import text_to_sentences, chunkise_sentences


def parse_text(raw_text: str, chunk_size: int = 1024):
    sentences = text_to_sentences(raw_text)
    return chunkise_sentences(sentences)
