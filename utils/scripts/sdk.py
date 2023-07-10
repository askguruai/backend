# -*- coding: utf-8 -*-
"""Tada demo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Feodoros/TadateamSubmission/blob/main/Tada%20demo.ipynb

# Globals

## You may want to play around with this
"""

import os

# v1 is faster, w/o footnote link;
# v2 slower, w/ footnote links (if possible)
API_VERSION = "v2"

"""## Probably not change these"""

# If True, the answer will be based only on collections in knowledge base.
# Otherwise, api will try to answer based on collections, but if it will not
# succeed it will generate answer from the model weights themselves.
# Everything happens under the hood, user don't have to specify anything.
COLLECTIONS_ONLY = False
API_URL = "https://api-dev1.askguru.ai"
VENDOR = "tada"
ORGANIZATION = "common"
TADA_PASSWORD = os.environ["TADA_PASSWORD"]
SOURCE_PATTERN = r"\{ *doc_idx *: *([^}]*)\}"
STREAM_ANSWER = True
IF_JUPYTER = True

"""## Collections names"""

COLLECTIONS_MAPPING = {
    "glavbuhqa": "Главбух: ответы на самые популярные вопросы за последний год",
    "glavbuharticles": "Главбух: лучшие статьи за последний год по налогам, страховым взносам и бухгалтерскому учету",
    "tgdulkarnaev": "Артур Дулкарнаев (налоговый юрист) — Telegram-канал",
    "nalog": "Налоговый кодекс Российской Федерации (НК РФ)",
    "zhilishchnyj": "Жилищный кодекс (ЖК РФ)",
    "KoAP": "Кодекс об административных правонарушениях (КоАП РФ)",
    "AO": 'Федеральный закон "Об АО"',
    "UK": "Уголовный кодекс (УК РФ)",
    "semejnyj": "Семейный кодекс (СК РФ)",
    "trud": "Трудовой кодекс (ТК РФ)",
    "tamozhnya": "Таможенный кодекс РФ от 28 мая 2003 г. N 61-ФЗ",
    "arbitr": "Арбитражный процессуальный кодекс (АПК РФ)",
    "zemlya": "Земельный кодекс (ЗК РФ)",
    "buhgalter": 'Федеральный закон "О бухгалтерском учете"',
    "grazhdan": "Гражданский кодекс Российской Федерации (ГК РФ)",
    "grad": "Градостроительный кодекс РФ (ГрК РФ) от 29 декабря 2004 г. N 190-ФЗ",
    "strahovanie": 'Федеральный закон от 29 декабря 2006 г. N 255-ФЗ "Об обязательном социальном страховании на случай временной нетрудоспособности и в связи с материнством" (с изменениями и дополнениями)',
    "grazhdanoprocesualny": "Гражданский процессуальный кодекс (ГПК РФ)",
    "konstitucia": "Конституция Российской Федерации",
    "personalnyedannye": 'Закон "О персональных данных"',
    "obrazovanie": "Закон об образовании",
    "voennyi": 'Федеральный закон "О воинской обязанности и военной службе"',
    "ohranazdorovia": 'Федеральный закон от 21 ноября 2011 г. N 323-ФЗ "Об основах охраны здоровья граждан в Российской Федерации" (с изменениями и дополнениями)',
    "potrebytely": 'Закон "О защите прав потребителей"',
    "goszakup": 'Закон о контрактной системе (закон о госзакупках). Федеральный закон от 5 апреля 2013 г. N 44-ФЗ "О контрактной системе в сфере закупок товаров, работ, услуг для обеспечения государственных и муниципальных нужд" (с изменениями и дополнениями)',
    "profilaktika": 'Федеральный закон от 24 июня 1999 г. N 120-ФЗ "Об основах системы профилактики безнадзорности и правонарушений несовершеннолетних" (с изменениями и дополнениями)',
    "pubvlast": 'Федеральный закон от 21 декабря 2021 г. N 414-ФЗ "Об общих принципах организации публичной власти в субъектах Российской Федерации" (с изменениями и дополнениями)',
}

"""## Imports"""


import json
import re
from pprint import pprint

import pandas as pd
import requests
from loguru import logger

"""# Api definition

These are just API calls with fancy results displaying
"""


class Api:
    def __init__(
        self,
        api_url: str,
        api_version: str,
        source_pattern: str,
        collections_mapping: dict = {},
        if_jupyter: bool = False,
    ):
        self.api_url = api_url
        self.api_version = api_version
        self.token = None
        self.if_jupyter = if_jupyter
        self.source_pattern = source_pattern
        self.collections_mapping = collections_mapping
        self.collections = []

    def _make_request(self, method: str, endpoint: str, params: dict = {}, json: dict = {}):
        url = f"{self.api_url}/{self.api_version}{endpoint}"
        response = requests.request(
            method,
            url,
            params=params,
            json=json,
            headers={"Authorization": f"Bearer {self.token}"},
            stream=params.get("stream", False),
        )
        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        return response

    def info(self):
        response = self._make_request("GET", "/info").json()
        logger.info(f"Token info: {response}")

    def authenticate(self, vendor: str, organization: str, password: str):
        json = {"vendor": vendor, "organization": organization, "password": password}
        response = self._make_request("POST", "/collections/token", json=json).json()
        self.token = response["access_token"]
        logger.info(f"Authenticated!")

    def get_collections(self):
        response = self._make_request("GET", "/collections").json()
        text = ""
        for collection in response["collections"]:
            text += f"- `{collection['name']}` ({collection['n_documents']} документов)"
            text += (
                f" — {self.collections_mapping[collection['name']]}"
                if collection['name'] in self.collections_mapping
                else ""
            )
            text += "\n"
        display(Markdown(text))
        self.collections = [collection['name'] for collection in response["collections"]]
        return response["collections"]

    def upload_collection_documents(self, collection: str, documents: list):
        json = {"documents": documents}
        response = self._make_request("POST", f"/collections/{collection}", json=json).json()
        logger.info(f"Uploaded documents: {response}")

    def get_ranking(self, query: str, collections: list[str] = [], top_k: int = 5):
        if not collections:
            collections = self.collections
        params = {"query": query, "collections": collections, "top_k": top_k}
        response = self._make_request("GET", "/collections/ranking", params=params).json()
        text = "Самые релевантные документы:\n"
        for i, source in enumerate(response["sources"]):
            text += f"{i + 1}. [{source['title'].replace('message text: ', '')}]({source['id']})"
            # text += f" ({source['summary'].split('>')[0]})" if '>' in source['summary'] else ""
            text += (
                f" ({self.collections_mapping[source['collection']]})"
                if source['collection'] in self.collections_mapping
                else ""
            )
            text += "\n"
        display(Markdown(text))
        return response["sources"]

    @staticmethod
    def postprocess_output(s: str) -> str:
        return s[: max(s.find(")"), s.rfind("."), s.rfind("?"), s.rfind("!"), s.rfind("\n")) + 1]

    def get_answer(
        self,
        query: str,
        user: str = "",
        collections: list[str] = [],
        collections_only: bool = False,
        stream: bool = False,
    ):
        if not collections:
            collections = self.collections
        params = {
            "collections": collections,
            "query": query,
            "user": user,
            "collections_only": collections_only,
            "stream": stream,
        }
        response = self._make_request("GET", f"/collections/answer", params=params)
        if stream:
            answer = ""
            generated_sources = []
            for line in response.iter_lines():
                if line.startswith(b'event: '):
                    event = line[len(b'event: ') :].decode()
                elif line.startswith(b'data: '):
                    data_str = line[len(b'data: ') :].decode()
                    data = json.loads(data_str)
                    if event == 'message':
                        sources = data['sources']
                        request_id = data['request_id']
                        answer += data['answer']
                        match = re.findall(self.source_pattern, answer)
                        if match:
                            idx = int(match[0])
                            source = sources[idx]
                            if source in generated_sources:
                                num = generated_sources.index(source) + 1
                            else:
                                generated_sources.append(source)
                                num = len(generated_sources)
                            answer = re.sub(self.source_pattern, f"[[{num}]]({source['id']})", answer)
                        if self.if_jupyter:
                            clear_output(wait=True)
                            display(Markdown(answer))
                        else:
                            print(answer, end='\r')
            answer = Api.postprocess_output(answer)
            clear_output(wait=True)
            display(Markdown(answer))
            if generated_sources:
                answer += "\n\n**Источники:**\n"
                for i, source in enumerate(generated_sources):
                    answer += f"{i + 1}. [{source['title'].replace('message text: ', '')}]({source['id']})\n"
                clear_output(wait=True)
                display(Markdown(answer))

            response = {"answer": answer, "sources": sources, "request_id": request_id}
        return response

    def set_reaction(self, request_id: str, rating: int):
        json = {"request_id": request_id, "rating": rating}
        response = self._make_request("POST", "/reactions", json=json)
        logger.info(response)

    def get_reactions(self):
        response = self._make_request("GET", "/reactions").json()
        response = pd.DataFrame(response["reactions"])
        response.sort_values(by="datetime", ascending=False, inplace=True)
        return response

    def get_filters(self):
        response = self._make_request("GET", "/filters").json()
        return response

    def upload_filter(self, name: str, stop_words: list[str], description: str = None):
        json = {"name": name, "description": description, "stop_words": stop_words}
        response = self._make_request("POST", "/filters", json=json)
        logger.info(response)

    def patch_filter(self, name: str, stop_words: list[str], description: str = None):
        json = {"name": name, "description": description, "stop_words": stop_words}
        response = self._make_request("PATCH", "/filters", json=json)
        logger.info(response)

    def archive_filter(self, name: str):
        json = name
        response = self._make_request("DELETE", "/filters", json=json)
        logger.info(response)


api = Api(
    api_url=API_URL,
    api_version=API_VERSION,
    source_pattern=SOURCE_PATTERN,
    collections_mapping=COLLECTIONS_MAPPING,
    if_jupyter=IF_JUPYTER,
)

"""## Retrieving token"""

api.authenticate(VENDOR, ORGANIZATION, TADA_PASSWORD)
api.info()


data = pd.read_csv("./data/tadahack/tg.csv")
documents = []
for text in data["Text"]:
    if text and type(text) == str:
        documents.append({"id": "https://t.me/arturdulkarnaev", "title": text.split("\r\n")[0], "content": text})
api.upload_collection_documents(collection="tgdulkarnaev", documents=documents)


"""## Getting collections"""


# collections = api.get_collections()

# """# Getting answers!

# Play around here, ask your own questions.

# You may want to specify collections to search in, but by default all available collections will be used.

# You may use `api.get_ranking` to retrieve most relevant docs although it is not mandatory for the answer.

# Note that API requests for questions which refer to existing knowledge and which use internal knowledge are the same.
# The model under the hood decides if it is appropriate to use knowledge from knowledge base.

# ## Questions which refer to existing knowledge
# """

# # you may specify any user identifier here
# USER = "322"

# """### Glavbuh QA"""

# query = "НДФЛ с иностранных граждан временно пребывающих на патенте, перечислять нужно отдельным платежом, как и в ФСС НС?"
# sources = api.get_ranking(query)

# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)
# api.set_reaction(request_id=response["request_id"], rating=5)

# """### Glavbuh Articles"""

# query = "Новые правила по отпускным 2023"
# sources = api.get_ranking(query)

# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)
# api.set_reaction(request_id=response["request_id"], rating=4)

# """### Telegram @dulkarnaev"""

# query = "штраф за неподачу отчета о движении денег"
# sources = api.get_ranking(query)

# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)
# api.set_reaction(request_id=response["request_id"], rating=5)

# """### Taxes"""

# query = "налоговые вычеты для студентов многодетных семей нк рф, коротко"
# sources = api.get_ranking(query)

# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)
# api.set_reaction(request_id=response["request_id"], rating=5)

# """### Labor"""

# query = "Исключается ли из оплаты сверхурочных работ работы, выполненные работником в выходной день?"
# sources = api.get_ranking(query)

# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)
# api.set_reaction(request_id=response["request_id"], rating=5)

# """## Common knowledge questions

# ### EBITDA
# """

# query = "как расчитывается ebitda"
# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)
# api.set_reaction(request_id=response["request_id"], rating=5)

# """### Design pricing"""

# query = "какова средняя стоимость дизайнера в час на рынке"
# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)
# api.set_reaction(request_id=response["request_id"], rating=4)

# """# Retrieving logs with feedback"""

# reactions = api.get_reactions()
# display(reactions)

# """# Filtering rules

# We do not send separate response that request has made it through filters and the answer is preparing.

# The reason for this is that we stream answer, so if you have not received an error right away, you will immediately receive an event stream with 200 status code, so there is no need in separate response.

# To every filter rule you may add description.
# """

# pprint(api.get_filters())

# query = "что значит мы русские, с нами бог"

# # 'бог' is not in ban
# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)

# api.patch_filter(name="religion", stop_words=["бог", "вера"])
# # 'бог' is in ban, so we are getting an error
# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)

# api.archive_filter(name="religion")
# # as rule is archived, we are able to query once again
# response = api.get_answer(query, user=USER, collections_only=COLLECTIONS_ONLY, stream=STREAM_ANSWER)

# # we can upload any new filter
# # api.upload_filter(name="cars", stop_words=["девятка", "жигули"])

# # patching archived filter leads to it resurrection
# api.patch_filter(name="religion", stop_words=["вера"])