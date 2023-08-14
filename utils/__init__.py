########################################################
#                       CONFIG                         #
########################################################
from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")


########################################################
#                       HASHING                        #
########################################################
import hashlib


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()[: int(CONFIG["misc"]["hash_size"])]


########################################################
#                       DATABASES                      #
########################################################
from utils.db import DB, GRIDFS
from utils.milvus_utils import CollectionsManager

MILVUS_DB = CollectionsManager(collections_cache_size=20)


########################################################
#                HTTP CONNECTIONS                      #
########################################################
from aiohttp import ClientSession


class ClientSessionWrapper:
    # It has to be set in async function
    # so we set in in main.py on startup
    coreml_session: ClientSession = None
    general_session: ClientSession = None


CLIENT_SESSION_WRAPPER = ClientSessionWrapper()


import os
from typing import List

########################################################
#                    AWS TRANSLATE                     #
########################################################
from utils.aws import AwsTranslateClient

AWS_TRANSLATE_CLIENT = AwsTranslateClient()
