from utils.tokenize_ import doc_to_chunks
from utils import MILVUS_DB, hash_string
from utils.milvus_utils import CollectionType
from utils import ml_requests
from utils.schemas import ApiVersion


async def upload_resources(vendor: str, organization: str, collection:str, content: str, doc_id: str, api_version: ApiVersion):
    org_hash = hash_string(organization)
    collection = MILVUS_DB.get_or_create_collection(collection_name=f"{vendor}_{org_hash}_{collection}",
                                                collection_type=CollectionType.EDUPLAT)
    chunks = doc_to_chunks(content=content)
    existing_chunks = collection.query(
        expr=f'doc_id=="{doc_id}"',
        offset=0,
        limit=10000,
        output_fields=["pk", "chunk_hash"],
        consistency_level="Eventually",
    )
    existing_chunks = {
        hit["chunk_hash"]: hit["pk"] for hit in existing_chunks
    }
    # determining which chunks are new
    new_chunks_hashes = []
    new_chunks = []
    for chunk in chunks:
        text_hash = hash_string(chunk)
        if text_hash in existing_chunks:
            del existing_chunks[text_hash]
        else:
            new_chunks_hashes.append(text_hash)
            new_chunks.append(chunk)
    # dropping outdated chunks
    existing_chunks_pks = list(existing_chunks.values())
    collection.delete(f"pk in [{','.join(existing_chunks_pks)}]")

    embeddings = await ml_requests.get_embeddings(chunks=new_chunks, api_version=api_version)
    collection.insert([
        new_chunks_hashes,
        [doc_id] * len(new_chunks_hashes),
        embeddings
    ])

    return {"n_chunks": len(new_chunks_hashes)}




