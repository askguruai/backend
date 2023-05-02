from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

from utils.db import DB
from utils.milvus_utils import CollectionsManager

MILVUS_DB = CollectionsManager(collections_cache_size=20)
