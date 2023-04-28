from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

from utils.db import DB

from milvus_db.utils import CollectionsManager
MILVUS_DB = CollectionsManager(collections_cache_size=20)
