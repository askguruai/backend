from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

import hashlib


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()[: int(CONFIG["misc"]["hash_size"])]


from utils.db import DB
from utils.milvus_utils import CollectionsManager

MILVUS_DB = CollectionsManager(collections_cache_size=20)


from aiohttp import ClientSession


class ClientSessionWrapper:
    # It has to be set in async function
    # so we set in in main.py on startup
    coreml_session: ClientSession = None
    general_session: ClientSession = None


CLIENT_SESSION_WRAPPER = ClientSessionWrapper()

from google.cloud import translate_v2 as translate

TRANSLATE_CLIENT = translate.Client()
