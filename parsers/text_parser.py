from parsers.common import chunkise_sentences, text_to_sentences


def parse_text(raw_text: str, chunk_size: int):
    sentences = text_to_sentences(raw_text)
    chunks = chunkise_sentences(sentences, chunk_size)
    return chunks
