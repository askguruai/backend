import html2text
import requests
import wikipedia
from wikipedia.exceptions import PageError

CONVERTER = html2text.HTML2Text()
CONVERTER.ignore_links = True

WIKI_LANGUAGES = ["en", "ru"]


def extract_text_from_link(link: str) -> str:
    text = parse_wiki(link) if "wikipedia.org" in link else ""
    return text if text else parse_website(link)


def parse_wiki(link: str) -> str:
    title = link.split('/')[-1]
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


def parse_website(link: str) -> str:
    r = requests.get(link)
    text = CONVERTER.handle(r.text)
    return text
