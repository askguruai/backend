import html2text
import requests
import wikipedia
from wikipedia.exceptions import PageError

from parsers.general_parser import GeneralParser

WIKI_LANGUAGES = ["en", "ru"]


class LinkParser(GeneralParser):
    def __init__(self, chunk_size: int):
        super().__init__(chunk_size=chunk_size)
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True

    def get_text(self, link: str) -> str:
        text = self.parse_wiki(link) if "wikipedia.org" in link else ""
        return text if text else self.parse_website(link)

    def parse_wiki(self, link: str) -> str:
        title = link.split("/")[-1]
        i, text = 0, ""
        while not text and i < len(WIKI_LANGUAGES):
            wikipedia.set_lang(WIKI_LANGUAGES[i])
            try:
                page = wikipedia.WikipediaPage(title=title)
                text = page.content
            except PageError:
                pass
            i += 1
        return text

    def parse_website(self, link: str) -> str:
        r = requests.get(link)
        text = self.converter.handle(r.text)
        return text
