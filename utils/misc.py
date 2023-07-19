import html2text
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from loguru import logger

from utils.errors import SecurityGroupError

HTML2TEXT = html2text.HTML2Text()
HTML2TEXT.ignore_images = True


def int_list_encode(group_list: list) -> int:
    if len(group_list) == 0:
        return 2**63 - 1
    n = 0
    for gr in group_list:
        if gr < 0 or gr > 63:
            raise SecurityGroupError(f"Invalid security group code: {gr}. Value must be between 0 and 63")
        n |= 1 << gr
    return n


async def parse_link(link: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    }
    async with ClientSession() as session:
        try:
            async with session.get(link, headers=headers) as response:
                page_content = await response.text()
        except Exception as e:
            logger.error(f"Error while downloading {link}: {e}")
            return None

    if page_content and page_content.count("<html") > 0:
        content = HTML2TEXT.handle(page_content)
        return content
    return None
