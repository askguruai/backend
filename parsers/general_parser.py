import abc
from typing import Any, List

from nltk import tokenize


class GeneralParser:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def get_chunks_from_text(self, text: str) -> List[str]:
        sentences = GeneralParser.text_to_sentences(text)
        chunks = GeneralParser.chunkise_sentences(sentences, self.chunk_size)
        return chunks

    @staticmethod
    def text_to_sentences(text: str) -> List[str]:
        return tokenize.sent_tokenize(text)

    @staticmethod
    def chunkise_sentences(sentences: List[str], chunk_size: int) -> List[str]:
        chunks = []
        current_chunk = ""
        sent_stack = sentences[::-1]
        while len(sent_stack) > 0:
            sent = sent_stack.pop()
            if len(sent) > chunk_size:
                shards = GeneralParser.split_long_sentence(sent, chunk_size)
                for shard in shards[::-1]:
                    sent_stack.append(shard)
            if len(current_chunk) + len(sent) <= chunk_size:
                current_chunk += sent
            else:
                chunks.append(current_chunk)
                current_chunk = sent
        chunks.append(current_chunk)
        return chunks

    @staticmethod
    def split_long_sentence(sentence: str, chunk_size: int) -> List[str]:
        if len(sentence) < chunk_size:
            return [sentence]
        words = sentence.split(" ")
        if len(words) < 2:
            return []  # omitting the sentence
        mid_space = len(words) // 2
        return GeneralParser.split_long_sentence(
            " ".join(words[:mid_space]), chunk_size
        ) + GeneralParser.split_long_sentence(" ".join(words[mid_space:]), chunk_size)

    @abc.abstractmethod
    def get_text(self, content: Any) -> str:
        return
