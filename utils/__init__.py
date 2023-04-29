from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

from utils.milvus_utils import CollectionsManager
from utils.db import DB

MILVUS_DB = CollectionsManager(collections_cache_size=20)
