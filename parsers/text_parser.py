from parsers.common import text_to_sentences, chunkise_sentences


def parse_text(raw_text: str, chunk_size: int):
    sentences = text_to_sentences(raw_text)
    chunks = chunkise_sentences(sentences, chunk_size)
    return chunks
