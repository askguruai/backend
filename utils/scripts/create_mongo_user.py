# pip install "typer[all]"

import os
import sys

sys.path.insert(1, os.getcwd())

from configparser import ConfigParser

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

import urllib

import pymongo
import typer
from loguru import logger

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


# from typing_extensions import Annotated

# import string
# import random
# def random_generator(size=12, chars=string.ascii_lowercase + string.digits):
#     return ''.join(random.choice(chars) for _ in range(size))


def create_user(
    vendor: str,
):
    username = password = role_name = vendor

    collections = [
        f"{vendor}.{CONFIG['mongo']['requests_collection']}",
        f"{vendor}.{CONFIG['mongo']['client_event_log_collection']}",
    ]

    privileges = [
        {
            "resource": {"db": DATABASE, "collection": collection},
            "actions": ["find", "update", "insert", "remove"],
        }
        for collection in collections
    ]

    DB.command("createRole", role_name, privileges=privileges, roles=[])
    logger.info(
        f"Successfully created role '{role_name}' with access to collections '{collections}' withing db '{DATABASE}'"
    )

    create_user_command = {
        "createUser": username,
        "pwd": password,
        "roles": [
            {
                "role": role_name,
                "db": DATABASE,
            },
        ],
        "customData": {"createdBy": "admin"},
    }

    DB.command(create_user_command)
    logger.info(f"Successfully created user '{username}' with password '{password}' and role '{role_name}'")


if __name__ == "__main__":
    typer.run(create_user)
