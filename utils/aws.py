import os
from typing import List

import boto3
from loguru import logger

boto_session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ["AWS_REGION"],
)


# NB! We can try to make this async with https://pypi.org/project/aioboto3/, but this is not an official library
class AwsTranslateClient:
    def __init__(self) -> None:
        self.translate_client = boto_session.client("translate")
        self.comprehend_client = boto_session.client("comprehend")

    def translate_text(self, text: str | List[str], target_language: str = "en", source_language: str = "auto") -> dict:
        source_lines = None
        if isinstance(text, list):
            source_lines = len(text)
            text = "\n***###***\n".join(text)

        if source_language == "auto":
            detection = self.comprehend_client.detect_dominant_language(Text=text.ljust(20)[:300])
            primary_lang = detection["Languages"][0]
            print(primary_lang)
            if primary_lang["Score"] > 0.8:
                source_language = primary_lang["LanguageCode"]
            else:
                # there is some weird text or terms or whatever, better not translate and leave it to the model
                if source_lines is not None:
                    text = [line.strip() for line in text.split("***###***")]
                return {"translation": text, "source_language": "en"}

        if source_language == target_language:
            if source_lines is not None:
                text = [line.strip() for line in text.split("***###***")]
            return {"translation": text, "source_language": source_language}

        # recursion
        if len(text) < 9900:
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
