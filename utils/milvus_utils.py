import os
from typing import Dict, List, Tuple

import numpy as np
from fastapi import HTTPException, status
from loguru import logger
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from utils import CONFIG, hash_string
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
            if collection_name.endswith("_website"):
                # not loading these to cache on startup
                continue
            col = Collection(collection_name)
            col.load()
            self.cache[collection_name] = col

    def get_collections(self, vendor: str, org_hash: str) -> List[Dict[str, int]]:
        collections = []
        for collection_name in utility.list_collections():
            if collection_name.startswith(f"{vendor}_{org_hash}_"):
                collections.append(
                    {
                        "name": collection_name.split("_")[-1],
                        "n_documents": self.get_collection(collection_name).num_entities,
                    }
                )
        return collections

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
        self,
        vendor: str,
        organization: str,
        collections: List[str],
        vec: np.ndarray,
        n_top: int,
        api_version: str,
        document_id_to_exclude: str = None,
        document_collection: str = None,
        security_code: int = 2**63 - 1,  # full access by default
    ) -> Tuple[List[str]]:
        org_hash = hash_string(organization)
        # search_collections = [self[col] for col in collections]
        # collections = [f"{vendor}_{org_hash}_{collection}" for collection in collections]
        search_collections = []
        for collection in collections:
            collection_name = f"{vendor}_{org_hash}_{collection}"
            try:
                search_collections.append(self.get_collection(collection_name))
            except DatabaseError as e:
                logger.error(
                    f"Requested collection '{collection}' not found in vendor '{vendor}' and organization '{organization}'! Organization hash: {org_hash}"
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Requested collection '{collection}' not found in vendor '{vendor}' and organization '{organization}'!",
                )

        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }
        all_chunks = []
        all_similarities = []
        all_titles = []
        all_ids = []
        all_summaries = []
        all_collections = []
        for i in range(1 + int(CONFIG["misc"]["collections_search_retries"])):
            for collection in search_collections:
                results = collection.search(
                    [vec],
                    f"emb_v1",
                    # f"emb_{api_version}",
                    search_params,
                    offset=i * int(CONFIG["misc"]["collections_search_limit"]),
                    limit=int(CONFIG["misc"]["collections_search_limit"]),
                    output_fields=["chunk", "doc_title", "doc_id", "doc_summary", "security_groups"],
                )[0]
                for dist, hit in zip(results.distances, results):
                    if (
                        hit.entity.get("doc_id") != document_id_to_exclude
                        or document_collection != collection.name.split("_")[-1]
                    ) and (hit.entity.get("security_groups") & security_code):
                        all_similarities.append(dist)
                        all_chunks.append(hit.entity.get("chunk"))
                        all_titles.append(hit.entity.get("doc_title"))
                        all_ids.append(hit.entity.get("doc_id"))
                        all_summaries.append(hit.entity.get("doc_summary"))
                        all_collections.append(collection.name)
            if len(all_chunks) >= n_top:
                # we found enough, can stop now
                break
        top_hits = np.argsort(all_similarities)[-n_top:][::-1]
        return (
            np.array(all_similarities)[top_hits].tolist(),
            np.array(all_chunks)[top_hits].tolist(),
            np.array(all_titles)[top_hits].tolist(),
            np.array(all_ids)[top_hits].tolist(),
            np.array(all_summaries)[top_hits].tolist(),
            np.array(all_collections)[top_hits].tolist(),
        )  # todo

    def get_or_create_collection(self, collection_name: str) -> Collection:
        if collection_name not in self.cache:
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(
                    name="chunk_hash",
                    dtype=DataType.VARCHAR,
                    max_length=24,
                ),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(
                    name="chunk", dtype=DataType.VARCHAR, max_length=int(CONFIG["milvus"]["chunk_max_symbols"])
                ),
                FieldSchema(name="emb_v1", dtype=DataType.FLOAT_VECTOR, dim=1536),
                FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="doc_summary", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="security_groups", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields)
            m_collection = Collection(collection_name, schema)
            index_params = {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            m_collection.create_index(field_name="emb_v1", index_params=index_params)
            m_collection.create_index(
                field_name="doc_id",
                index_name="scalar_index",
            )
            # todo: do we need an index on primary key? we do if it is not auto, need to check
            m_collection.load()
            self.cache[collection_name] = m_collection
        return self.cache[collection_name]
