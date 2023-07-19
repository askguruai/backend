from typing import List

from utils import MILVUS_DB, hash_string, ml_requests
from utils.milvus_utils import CollectionType
from utils.misc import parse_link
from utils.schemas import ApiCollection, ApiVersion, DescriptionRequest, SearchFilters, SearchResult
from utils.tokenize_ import doc_to_chunks


async def upload_resource(
    vendor: str,
    organization: str,
    content: str,
    doc_id: str,
    type: int,
    paid: int,
    difficulty: int,
    api_version: ApiVersion,
):
    org_hash = hash_string(organization)
    collection = MILVUS_DB.get_or_create_collection(
        collection_name=f"{vendor}_{org_hash}_resources", collection_type=CollectionType.EDUPLAT
    )
    chunks = doc_to_chunks(content=content)
    existing_chunks = collection.query(
        expr=f'doc_id=="{doc_id}"',
        offset=0,
        limit=10000,
        output_fields=["pk", "chunk_hash", "type", "paid", "difficulty"],
        consistency_level="Eventually",
    )
    existing_chunks = {hit["chunk_hash"]: (hit["pk"], hit["type"], hit["paid"], hit["difficulty"]) for hit in existing_chunks}
    # determining which chunks are new
    new_chunks_hashes = []
    new_chunks = []
    for chunk in chunks:
        text_hash = hash_string(chunk)
        if (
            text_hash in existing_chunks and
            existing_chunks[text_hash][1] == type and
            existing_chunks[text_hash][2] == paid and
            existing_chunks[text_hash][3] == difficulty
        ):
            del existing_chunks[text_hash]
        else:
            new_chunks_hashes.append(text_hash)
            new_chunks.append(chunk)
    # dropping outdated chunks
    existing_chunks_pks = list(map(lambda val: str(val[0]), existing_chunks.values()))
    collection.delete(f"pk in [{','.join(existing_chunks_pks)}]")

    if len(new_chunks) > 0:
        embeddings = await ml_requests.get_embeddings(chunks=new_chunks, api_version=api_version)
        collection.insert(
            [
                new_chunks_hashes,
                [doc_id] * len(new_chunks_hashes),
                embeddings,
                [type] * len(new_chunks_hashes),
                [paid] * len(new_chunks_hashes),
                [difficulty] * len(new_chunks_hashes),
            ]
        )

    return {"n_chunks": len(new_chunks_hashes)}


async def upload_topic(
    vendor: str,
    organization: str,
    content: str,
    doc_id: str,
    difficulty: int,
    api_version: ApiVersion
):
    org_hash = hash_string(organization)
    collection = MILVUS_DB.get_or_create_collection(
        collection_name=f"{vendor}_{org_hash}_topics", collection_type=CollectionType.EDUPLAT_TOPICS
    )
    chunks = doc_to_chunks(content=content)
    existing_chunks = collection.query(
        expr=f'doc_id=="{doc_id}"',
        offset=0,
        limit=10000,
        output_fields=["pk", "chunk_hash", "difficulty"],
        consistency_level="Eventually",
    )
    existing_chunks = {hit["chunk_hash"]: (hit["pk"], hit["difficulty"]) for hit in existing_chunks}
    # determining which chunks are new
    new_chunks_hashes = []
    new_chunks = []
    for chunk in chunks:
        text_hash = hash_string(chunk)
        if (
            text_hash in existing_chunks and
            existing_chunks[text_hash][1] == difficulty
        ):
            del existing_chunks[text_hash]
        else:
            new_chunks_hashes.append(text_hash)
            new_chunks.append(chunk)
    # dropping outdated chunks
    existing_chunks_pks = list(map(lambda val: str(val[0]), existing_chunks.values()))
    collection.delete(f"pk in [{','.join(existing_chunks_pks)}]")


    if len(new_chunks) > 0:
        embeddings = await ml_requests.get_embeddings(chunks=new_chunks, api_version=api_version)
        collection.insert(
            [
                new_chunks_hashes,
                [doc_id] * len(new_chunks_hashes),
                embeddings,
                [difficulty] * len(new_chunks_hashes),
            ]
        )

    return {"n_chunks": len(new_chunks_hashes)}


async def search(
    vendor: str,
    organization: str,
    query: str,
    filters: SearchFilters,
    api_version: ApiVersion,
    score_range: float = 0.07,
) -> SearchResult:
    query_embedding = (await ml_requests.get_embeddings(query, api_version.value))[0]
    resources = []
    topics = []
    for col in filters.collections:
        if col == ApiCollection.RESOURCES:
            org_hash = hash_string(organization)
            collection = MILVUS_DB[f"{vendor}_{org_hash}_{ApiCollection.RESOURCES}"]
            # todo: skip expressions if all types are checked!
            search_exps = [f"type in [{','.join(map(str, filters.types))}]"]
            if len(filters.paid) > 0:
                search_exps.append(f"paid in [{','.join(map(str, filters.paid))}]")
            if len(filters.difficulty) > 0:
                search_exps.append(f"difficulty in [{','.join(map(str, filters.difficulty))}]")
            search_expression = " and ".join(search_exps)
            search_param = {
                "data": [query_embedding],
                "anns_field": "emb_v1",
                "param": {"metric_type": "IP", "params": {"nprobe": 10}},
                "offset": 0,
                "limit": 10,
                "expr": search_expression,
                "output_fields": ["doc_id"],
            }
            result = collection.search(**search_param)[0]
            scores = result.distances
            print(scores, result.ids)
            top_score = scores[0]
            borderline = top_score * (1 - score_range)
            for i, hit in enumerate(result):
                if scores[i] >= borderline:
                    resources.append(hit.entity.get("doc_id"))
        if col == ApiCollection.TOPICS:
            pass
    return SearchResult(resources=resources, topics=topics)


async def describe_resource(desc_request: DescriptionRequest, api_version: ApiVersion) -> dict:
    content = await parse_link(desc_request.resource)
    if content is None:
        return {"status": "fail"}
    resource_description = await ml_requests.get_summary(
        info=content, max_tokens=desc_request.summary_length, api_version=api_version.value
    )
    return {"status": "ok", "description": resource_description}
