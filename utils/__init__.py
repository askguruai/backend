from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

from milvus_db.utils import CollectionsManager
from utils.db import DB

MILVUS_DB = CollectionsManager(collections_cache_size=20)
