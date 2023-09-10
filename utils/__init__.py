########################################################
#                       CONFIG                         #
########################################################
from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")


########################################################
#                   HASHING & NAMING                   #
########################################################
import hashlib


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode()).hexdigest()[: int(CONFIG["misc"]["hash_size"])]


def full_collection_name(vendor: str, organization: str, collection: str, is_canned=False, if_hash_org=False) -> str:
    organization = hash_string(organization) if if_hash_org else organization
    collection_name = f"{vendor}_{organization}_{collection}"
    return (
        collection_name if not is_canned else f"{collection_name}{CONFIG['milvus']['canned_answer_table_name_suffix']}"
    )


def get_collection_name(full_collection_name: str) -> str:
    return full_collection_name.split("_", maxsplit=3)[2]


########################################################
#                       DATABASES                      #
########################################################
from utils.db import DB, GRIDFS
from utils.milvus_utils import CollectionsManager

MILVUS_DB = CollectionsManager()


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
