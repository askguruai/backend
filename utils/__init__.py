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


########################################################
#                GOOGLE TRANSLATE                      #
########################################################
import json
import os

from google.cloud import translate_v2 as translate

GOOGLE_APPLICATION_CREDENTIALS = {
    "client_id": os.environ["GOOGLE_CLIENT_ID"],
    "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
    "quota_project_id": os.environ["GOOGLE_QUOTA_PROJECT_ID"],
    "refresh_token": os.environ["GOOGLE_REFRESH_TOKEN"],
    "type": "authorized_user",
}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./google_credentials.json"

with open("./google_credentials.json", "w") as f:
    json.dump(GOOGLE_APPLICATION_CREDENTIALS, f)

TRANSLATE_CLIENT = translate.Client()
