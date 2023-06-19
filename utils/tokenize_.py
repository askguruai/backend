from utils import CONFIG
import tiktoken
from typing import List


TOKEIZERS = {}

def get_tokenizer(tokenizer_name: str):
    if tokenizer_name not in TOKEIZERS:
        TOKEIZERS[tokenizer_name] = tiktoken.get_encoding(tokenizer_name)
    return TOKEIZERS[tokenizer_name]


def doc_to_chunks(content: str, title: str = "", summary: str = "",
                  tokenizer_name: str = CONFIG["handlers"]["tokenizer_name"],
                  chunk_size: int = int(CONFIG["handlers"]["chunk_size"])) -> List[str]:
        chunks = []
        encoder = get_tokenizer(tokenizer_name)

        # TODO: omit title because it is already in summary

        current_content, part = f"{summary}\n\n", 0
        # TODO: split by lines which are bolded (in case of cars)
        # because they are the titles of the sections
        for line in content.split("\n"):
            if len(encoder.encode(current_content + line + "\n")) > chunk_size:
                chunks.append(current_content.strip())
                current_content = f"{summary}\n\n"
                part += 1

            line_tokens = encoder.encode(line)
            if len(line_tokens) > chunk_size:
                line_token_parts = [
                    line_tokens[i : i + chunk_size] for i in range(0, len(line_tokens), chunk_size)
                ]

                for line_token_part in line_token_parts:
                    line_part = encoder.decode(line_token_part)
                    if len(encoder.encode(current_content + line_part + "\n")) > chunk_size:
                        chunks.append(current_content.strip())
                        current_content = f"{summary}\n\n"
                    current_content += line_part + "\n"

            else:
                current_content += line + "\n"

        chunks.append(current_content.strip())
        return chunks


# preload config tokenizer
TOKEIZERS[CONFIG["handlers"]["tokenizer_name"]] = get_tokenizer(CONFIG["handlers"]["tokenizer_name"])