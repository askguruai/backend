import os
from configparser import ConfigParser

import pytest
import requests

CONFIG = ConfigParser()
CONFIG.read("./config.ini")

if not "AUTH_COLLECTION_PASSWORD" in os.environ:
    print("Password not found in environment -> Probably running in github actions -> Using default password")
    with open(".env") as f:
        for line in f:
            if line.startswith('AUTH_COLLECTION_PASSWORD='):
                os.environ["AUTH_COLLECTION_PASSWORD"] = line.strip().split('=', 1)[1]


@pytest.fixture(name="manager", scope="class")
def manager_fixture():
    class Manager:
        def __init__(self):
            pass

    return Manager()


class TestAPI:
    BASE_URL = f"http://{CONFIG['app']['host']}:{CONFIG['app']['port']}"
    API_VERSION = "v1"

    def test_get_token(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/token"
        data = {
            "vendor": "askguru",
            "organization": "mcdonalds",
            "password": os.getenv("AUTH_COLLECTION_PASSWORD"),
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        manager.token = response.json().get('access_token')
        assert manager.token, "Failed to get token"

    def test_get_info(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/info"
        headers = {"Authorization": f"Bearer {manager.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        assert response.json()["vendor"] == "askguru"
        assert response.json()["organization"] == "mcdonalds"
        assert response.json()["security_groups"] == []

    def test_retrieve_empty(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections"
        headers = {"Authorization": f"Bearer {manager.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        assert response.json()["collections"] == []

    def test_upload_docs(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/recipes/docs"
        headers = {"Authorization": f"Bearer {manager.token}"}
        json = {
            "documents": [
                {
                    "content": "The Big Mac recipe consists of two all-beef patties, special sauce, lettuce, cheese, pickles, onions, sandwiched between a three-part sesame seed bun."
                },
                {
                    "content": "Wrap seasoned grilled chicken, lettuce, tomato, and a creamy sauce in a soft flour tortilla for a delicious and easy-to-make Chicken Twister."
                },
            ],
            "metadata": [
                {"id": 228, "title": "Big Mac recipe", "project_to_en": False},
                {"id": 322, "title": "Twister recipe", "project_to_en": False},
            ],
        }
        response = requests.post(url, headers=headers, json=json)
        response.raise_for_status()
        manager.test_upload_docs_chunks_inserted = int(response.json()["n_chunks"])
        assert manager.test_upload_docs_chunks_inserted > 0

    def test_retrieve_collections(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections"
        headers = {"Authorization": f"Bearer {manager.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        assert response.json()["collections"][0]["name"] == "recipes"
        assert response.json()["collections"][0]["n_chunks"] == manager.test_upload_docs_chunks_inserted

    def test_remove_collection(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/recipes"
        headers = {"Authorization": f"Bearer {manager.token}"}
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        assert int(response.json()["n_chunks"]) == manager.test_upload_docs_chunks_inserted
