from configparser import ConfigParser
from utils.storage_dispatcher import StorageDispatcher

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

STORAGE = StorageDispatcher(CONFIG["app"]["storage_path"])
