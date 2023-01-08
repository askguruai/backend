import os
import urllib

import pymongo
import gridfs

from utils import CONFIG

HOST = CONFIG["mongo"]["host"]
PORT = CONFIG["mongo"]["port"]
DATABASE = CONFIG["mongo"]["db"]
USER = os.environ["MONGO_INITDB_ROOT_USERNAME"]
PASSWORD = urllib.parse.quote_plus(os.environ["MONGO_INITDB_ROOT_PASSWORD"])

mongo_client = pymongo.MongoClient(
    host=str(HOST) + ":" + str(PORT),
    serverSelectionTimeoutMS=3000,
    username=USER,
    password=PASSWORD,
)

DB = mongo_client[DATABASE]
GRIDFS = gridfs.GridFS(DB)
