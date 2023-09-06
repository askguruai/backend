import asyncio
import json
import os
from configparser import ConfigParser

import pytest
import requests
from aiohttp import ClientSession

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

    ################################################################
    #                       AUTHORIZATION                          #
    ################################################################

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

    ################################################################
    #                       UPLOADING                              #
    ################################################################

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
                    "content": "The Big Mac recipe consists of two all-beef patties, special sauce, lettuce, cheese, pickles, onions, sandwiched between a three-part sesame seed bun. Bob ate eight of those."
                },
                {
                    "content": "Wrap seasoned grilled chicken, lettuce, tomato, and a creamy sauce in a soft flour tortilla for a delicious and easy-to-make Chicken Twister."
                },
            ],
            "metadata": [
                {"id": 228, "title": "Big Mac recipe"},
                {"id": 322, "title": "Twister recipe"},
            ],
        }
        response = requests.post(url, headers=headers, json=json)
        response.raise_for_status()
        manager.test_chunks_inserted = int(response.json()["n_chunks"])
        assert manager.test_chunks_inserted > 0

    def test_upload_pdf(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/recipes/files"
        headers = {"Authorization": f"Bearer {manager.token}"}
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_files_directory = os.path.join(dir_path, "files")
        manager.test_file_custom_summary = "Some custom summary"
        raw_documents = [
            ('files', open(os.path.join(test_files_directory, "Brief_Summary.pdf"), 'rb')),
        ]
        data = {
            "metadata": json.dumps(
                [
                    {"id": "pdf_file", "title": "Brief Summary", "summary": manager.test_file_custom_summary},
                ]
            )
        }
        response = requests.post(url, headers=headers, files=raw_documents, data=data)
        response.raise_for_status()
        assert int(response.json()["n_chunks"]) > 0
        manager.test_chunks_inserted += int(response.json()["n_chunks"])

    def test_upload_links(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/recipes/links"
        headers = {"Authorization": f"Bearer {manager.token}"}
        json = {
            "links": ["https://www.askguru.ai/", "https://yuma.ai/sitemap.xml"],
        }
        response = requests.post(url, headers=headers, json=json)
        response.raise_for_status()
        assert int(response.json()["n_chunks"]) > 0
        manager.test_chunks_inserted += int(response.json()["n_chunks"])

    def test_retrieve_collections(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections"
        headers = {"Authorization": f"Bearer {manager.token}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        assert response.json()["collections"][0]["name"] == "recipes"
        assert response.json()["collections"][0]["n_chunks"] == manager.test_chunks_inserted

    ################################################################
    #                       ANSWERING                              #
    ################################################################

    def test_get_answer(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "How many Big Macs did Bob ate?"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        manager.request_id = response.json()["request_id"]
        assert "eight" in response.json()["answer"].lower() or "8" in response.json()["answer"].lower()
        assert "228" in [source["id"] for source in response.json()["sources"]]

    def test_get_answer_stream(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "How many Big Macs did Bob ate?", "stream": True}
        response = requests.get(url, headers=headers, params=params, stream=True)
        response.raise_for_status()
        answer = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                k, v = line.split(':', 1)
                if k == 'data':
                    data = json.loads(v.strip())
                    answer += data['answer']
                    request_id = data['request_id']
                    sources = data['sources']
        answer = answer.lower()
        assert "eight" in answer or "8" in answer
        assert "228" in [source["id"] for source in sources]

    def test_get_answer_translation(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "Каково решение проблемы поиска ресурсов для обучения?"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        assert "платформа" in response.json()["answer"].lower()
        sources_ids = [source["id"] for source in response.json()["sources"]]
        assert "pdf_file" in sources_ids
        pdf_source = response.json()["sources"][sources_ids.index("pdf_file")]
        assert pdf_source["summary"] == manager.test_file_custom_summary

    def test_get_answer_translation_stream(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "Каково решение проблемы поиска ресурсов для обучения?", "stream": True}
        response = requests.get(url, headers=headers, params=params, stream=True)
        response.raise_for_status()
        answer = ""
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                k, v = line.split(':', 1)
                if k == 'data':
                    data = json.loads(v.strip())
                    answer += data['answer']
                    request_id = data['request_id']
                    sources = data['sources']
        answer = answer.lower()
        assert "платформа" in answer
        sources_ids = [source["id"] for source in sources]
        assert "pdf_file" in sources_ids
        pdf_source = sources[sources_ids.index("pdf_file")]
        assert pdf_source["summary"] == manager.test_file_custom_summary

    def test_get_answer_translation2(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "What is the solution to a problem of finding educational resources?"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        assert "platform" in response.json()["answer"].lower()
        sources_ids = [source["id"] for source in response.json()["sources"]]
        assert "pdf_file" in sources_ids
        pdf_source = response.json()["sources"][sources_ids.index("pdf_file")]
        assert pdf_source["summary"] == manager.test_file_custom_summary

    def test_get_answer_from_parsed_website(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "where askguru is registred?"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        assert "94121" in response.json()["answer"].lower() or "california" in response.json()["answer"].lower()

    def test_get_answer_from_xml_parsed_website(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "who is the founder and ceo of Yuma?"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        assert "guillaume" in response.json()["answer"].lower() or "luccisano" in response.json()["answer"].lower()

    def test_get_answer_chat(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {
            "chat": json.dumps(
                [
                    {"role": "user", "content": "whats the pricing for askguru"},
                    {
                        "role": "assistant",
                        "content": "Most common is the scale tier which begins with $5/mo and supports up to 10k workspaces and 1k docs.",
                    },
                    {"role": "user", "content": "cheapest option"},
                ]
            ),
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        manager.chat_request_id = response.json()["request_id"]
        assert "3" in response.json()["answer"].lower()

    def test_get_answer_chat_translation(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {
            "chat": json.dumps(
                [
                    {"role": "user", "content": "С какой проблемой сталкиваются люди, обучающиеся в интернете?"},
                    {
                        "role": "assistant",
                        "content": "С проблемой отсутствия доверия к ресурсам и сложности выбора подходящего ресурса",
                    },
                    {"role": "user", "content": "Как решить эту проблему?"},
                ]
            ),
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        assert "платформ" in response.json()["answer"].lower()

    def test_fix_answer(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/recipes/fix_answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        json = {"request_id": manager.request_id, "answer": "Bob ate 8 Big Macs and 1 Big Mac after, total of 9"}
        response = requests.post(url, headers=headers, json=json)
        response.raise_for_status()
        assert int(response.json()["n_chunks"]) > 0
        manager.test_chunks_inserted += int(response.json()["n_chunks"])

    def test_fix_answer_after_fix(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {"query": "How many Big Macs did Bob ate?"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        assert "9" in response.json()["answer"].lower() or "nine" in response.json()["answer"].lower()
        assert manager.request_id in [source["id"] for source in response.json()["sources"]]

    def test_fix_answer_chat(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/recipes/fix_answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        json = {"request_id": manager.chat_request_id, "answer": "Cheapest option for AskGuru is $2/mo"}
        response = requests.post(url, headers=headers, json=json)
        response.raise_for_status()
        assert int(response.json()["n_chunks"]) > 0
        manager.test_chunks_inserted += int(response.json()["n_chunks"])

    def test_fix_answer_chat_after_fix(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/answer"
        headers = {"Authorization": f"Bearer {manager.token}"}
        params = {
            "chat": json.dumps(
                [
                    {"role": "user", "content": "whats the pricing for askguru"},
                    {
                        "role": "assistant",
                        "content": "Most common is the scale tier which begins with $5/mo and supports up to 10k workspaces and 1k docs.",
                    },
                    {"role": "user", "content": "cheapest option"},
                ]
            ),
        }
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        assert "2" in response.json()["answer"].lower()

    ################################################################
    #                       CLEANING UP                            #
    ################################################################

    def test_remove_collection(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/collections/recipes"
        headers = {"Authorization": f"Bearer {manager.token}"}
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        assert int(response.json()["n_chunks"]) == manager.test_chunks_inserted

    async def __delete_async_request(self, session: ClientSession, url: str, params: dict = {}, headers: dict = {}):
        async with session.delete(url, headers=headers, params=params) as response:
            return await response.json(), response.status

    def test_absent_collection(self, manager):
        async def _multi_requests_404(headers: dict = {}, num_requests=5):
            session = ClientSession(f"{self.BASE_URL}")
            results = await asyncio.gather(
                *[
                    self.__delete_async_request(session, f"/{self.API_VERSION}/collections/recipes", headers=headers)
                    for _ in range(num_requests)
                ]
            )
            statuses = [res[1] for res in results]
            assert statuses == [404] * num_requests

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            asyncio.run(_multi_requests_404(headers={"Authorization": f"Bearer {manager.token}"}))
        finally:
            loop.close()

    def test_client_event(self, manager):
        url = f"{self.BASE_URL}/{self.API_VERSION}/events"
        headers = {"Authorization": f"Bearer {manager.token}"}
        json = {"type": "TEST", "context": {"hint": "This is a test event", "data": [1, 2, 3]}}
        response = requests.post(url, headers=headers, json=json)
        response.raise_for_status()
