from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

from utils.db import DB
from utils.milvus_utils import CollectionsManager

MILVUS_DB = CollectionsManager(collections_cache_size=20)

import hashlib


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()[: int(CONFIG["misc"]["hash_size"])]
