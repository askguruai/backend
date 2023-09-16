import re

from utils.errors import SecurityGroupError


def int_list_encode(group_list: list | None) -> int:
    if group_list is None or len(group_list) == 0:
        return 2**63 - 1
    n = 0
    for gr in group_list:
        if gr < 0 or gr > 63:
            raise SecurityGroupError(f"Invalid security group code: {gr}. Value must be between 0 and 63")
        n |= 1 << gr
    return n


def decode_security_code(n):
    if n == 2**63 - 1:
        return None  # idk if that's misleading though
    return [i for i, b in enumerate(bin(n)[:1:-1]) if b == "1"]


class AsyncIterator:
    def __init__(self, list):
        self.list = list
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.list):
            result = self.list[self.index]
            self.index += 1
            return result
        else:
            raise StopAsyncIteration


def romanize_hindi(input_text):
    sym = input_text
    replacements = {
        "क": "k",
        "ख": "kh",
        "ग": "ga",
        "घ": "gh",
        "ङ": "ng",
        "च": "ch",
        "छ": "chh",
        "ज": "j",
        "झ": "jh",
        "ञ": "ny",
        "ट": "t",
        "ठ": "th",
        "ड": "d",
        "ढ": "dh",
        "ण": "n",
        "त": "t",
        "थ": "th",
        "द": "d",
        "ध": "dh",
        "न": "n",
        "प": "p",
        "फ": "f",
        "ब": "b",
        "भ": "bh",
        "म": "m",
        "य": "y",
        "र": "r",
        "ल": "l",
        "व": "v",
        "श": "sh",
        "ष": "s",
        "स": "s",
        "ह": "h",
        "क़": "k",
        "ख़": "kh",
        "ग़": "g",
        "ऩ": "n",
        "ड़": "d",
        "ढ": "dh",
        "ढ़": "rh",
        "ऱ": "r",
        "य़": "ye",
        "ळ": "l",
        "ऴ": "ll",
        "फ़": "f",
        "ज़": "z",
        "ऋ": "ri",
        "ा": "aa",
        "ि": "i",
        "ी": "i",
        "ु": "u",
        "ू": "u",
        "ॅ": "e",
        "ॆ": "e",
        "े": "e",
        "ै": "ai",
        "ॉ": "o",
        "ॊ": "o",
        "ो": "o",
        "ौ": "au",
        "अ": "a",
        "आ": "aa",
        "इ": "i",
        "ई": "ee",
        "उ": "u",
        "ऊ": "oo",
        "ए": "e",
        "ऐ": "ai",
        "ऑ": "au",
        "ओ": "o",
        "औ": "au",
        "ँ": "n",
        "ं": "n",
        "ः": "ah",
        "़": "e",
        "्": "",
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
        "।": ".",
        "ऍ": "e",
        "ृ": "ri",
        "ॄ": "rr",
        "ॠ": "r",
        "ऌ": "l",
        "ॣ": "l",
        "ॢ": "l",
        "ॡ": "l",
        "ॿ": "b",
        "ॾ": "d",
        "ॽ": "",
        "ॼ": "j",
        "ॻ": "g",
        "ॐ": "om",
        "ऽ": "'",
        "e.a": "a",
        "\n": "\n",
    }

    for k, v in replacements.items():
        sym = re.sub(k, v, sym)

    return sym
