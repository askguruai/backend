import os
from collections import deque
from typing import List

import boto3
from loguru import logger

from utils import CONFIG
from utils.errors import TranslationError
from utils.schemas import Message, Role

boto_session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
)


def utf8len(s: str) -> int:
    return len(s.encode("utf-8"))


# NB! We can try to make this async with https://pypi.org/project/aioboto3/, but this is not an official library
class AwsTranslateClient:
    def __init__(self) -> None:
        self.translate_client = boto_session.client("translate")
        self.comprehend_client = boto_session.client("comprehend")

    def translate_chat(self, chat: List[Message], source_language="auto", target_language="en") -> List[Message]:
        user_last = deque(maxlen=3)
        chat_contents = []
        for msg in chat:
            if msg.role == Role.user:
                user_last.appendleft(msg.content)
            chat_contents.append(msg.content)
        if source_language == "auto":
            source_language = self.detect_language("\n".join(user_last))
            if source_language is None:
                source_language = "en"
        if source_language == target_language:
            return chat, source_language

        translated_contents = self.translate_text(
            chat_contents, source_language=source_language, target_language=target_language
        )["translation"]
        if len(chat_contents) != len(translated_contents):
            raise TranslationError()
        trans_chat = []
        for i in range(len(chat)):
            trans_chat.append(Message(role=chat[i].role, content=translated_contents[i]))
        return trans_chat, source_language

    def detect_language(self, text: str) -> str:
        detection = self.comprehend_client.detect_dominant_language(Text=text.ljust(20)[:300])
        primary_lang = detection["Languages"][0]
        if primary_lang["Score"] > float(CONFIG["misc"]["language_detection_min_confidence"]):
            return primary_lang["LanguageCode"]
        else:
            # unable to confidently predict language
            return None

    def translate_text(self, text: str | List[str], target_language: str = "en", source_language: str = "auto") -> dict:
        source_lines = None
        if isinstance(text, list):
            source_lines = len(text)
            text = "\n***###***\n".join(text)

        if source_language == "auto":
            source_language = self.detect_language(text)
            if source_language is None:
                # there is some weird text or terms or whatever, better not translate and leave it to the model
                source_language = "en"

        if source_language == target_language:
            if source_lines is not None:
                text = [line.strip() for line in text.split("***###***")]
            return {"translation": text, "source_language": source_language}

        logger.info(
            f"Translating from {source_language} into {target_language}\nText: '{text[:100]}' (total length {len(text)})"
        )

        # recursion
        if utf8len(text) < 10000:
            response = self.translate_client.translate_text(
                Text=text, SourceLanguageCode=source_language, TargetLanguageCode=target_language
            )
            out = {"translation": response["TranslatedText"], "source_language": response["SourceLanguageCode"]}
        else:
            lines = text.split("\n")
            if len(lines) < 2:
                # wtf is this text...
                divider = len(text) // 2
                part1 = text[:divider]
                part2 = text[divider:]
            else:
                divider = len(lines) // 2
                part1 = "\n".join(lines[:divider])
                part2 = "\n".join(lines[divider:])
            translation1 = self.translate_text(part1, target_language=target_language, source_language=source_language)
            translation2 = self.translate_text(part2, target_language=target_language, source_language=source_language)
            if translation1["source_language"] != translation2["source_language"]:
                logger.error(f"Parts language detected to be different! Idk what to do about it tho, just logging")
            out = {
                "translation": f"{translation1['translation']}\n{translation2['translation']}",
                "source_language": translation2["source_language"],
            }

        if source_lines is not None:
            # we should split them back
            text_lines = [line.strip() for line in out["translation"].split("***###***")]
            assert len(text_lines) == source_lines, (len(text_lines), source_lines)
            out["translation"] = text_lines
        return out
