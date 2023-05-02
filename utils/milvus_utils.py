import os
from typing import List, Tuple

import numpy as np
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from utils import CONFIG
from utils.errors import DatabaseError

connections.connect(
    "default",
    host=CONFIG["milvus"]["host"],
    port=CONFIG["milvus"]["port"],
    user=os.environ["MILVUS_USERNAME"],
    password=os.environ["MILVUS_PASSWORD"],
)


class CollectionsManager:
    def __init__(self, collections_cache_size=20) -> None:
        self.cache_size = collections_cache_size  # does not do anything just yet
        self.cache = {}
        for collection_name in utility.list_collections():
            col = Collection(collection_name)
            col.load()
            self.cache[collection_name] = col

    def get_collection(self, collection_name: str) -> Collection:
        if collection_name not in self.cache:
            if collection_name not in utility.list_collections():
                raise DatabaseError(f"Colletion {collection_name} not found!")
            else:
                m_collection = Collection(collection_name)
                m_collection.load()
                self.cache[collection_name] = m_collection
        return self.cache[collection_name]

    def __getitem__(self, name: str) -> Collection:
        return self.get_collection(name)

    def search_collections_set(
        self, collections: List[str], vec: np.ndarray, n_top: int, api_version: str
    ) -> Tuple[List[str]]:
        search_collections = [self[col] for col in collections]
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }
        all_chunks = []
        all_distances = []
        all_titles = []
        all_ids = []
        all_summaries = []
        for collection in search_collections:
            results = collection.search(
                [vec],
                f"emb_{api_version}",
                search_params,
                limit=n_top,
                output_fields=["chunk", "doc_title", "doc_id", "doc_summary"],
            )[0]
            all_distances.extend(results.distances)
            for hit in results:
                all_chunks.append(hit.entity.get("chunk"))
                all_titles.append(hit.entity.get("doc_title"))
                all_ids.append(hit.entity.get("doc_id"))
                all_summaries.append(hit.entity.get("doc_summary"))
        top_hits = np.argsort(all_distances)[-n_top:]
        return (
            np.array(all_chunks)[top_hits].tolist(),
            np.array(all_titles)[top_hits].tolist(),
            np.array(all_ids)[top_hits].tolist(),
            np.array(all_summaries)[top_hits].tolist(),
        )  # todo

    def get_or_create_collection(self, collection_name: str) -> Collection:
        if collection_name in self.cache:
            return self.cache[collection_name]
        fields = [
            FieldSchema(
                name="chunk_hash",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=24,
            ),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="emb_v1", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="doc_summary", dtype=DataType.VARCHAR, max_length=2048),
        ]
        schema = CollectionSchema(fields)
        m_collection = Collection(collection_name, schema)
        index_params = {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        m_collection.create_index(field_name="emb_v1", index_params=index_params)
        # todo: do we need an index on primary key? we do if it is not auto, need to check
        m_collection.load()
        self.cache[collection_name] = m_collection
        return m_collection
