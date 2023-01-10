from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

from utils.db import DB
