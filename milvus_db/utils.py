from pymilvus import (
    connections,
    utility as m_utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

connections.connect("default", host="localhost", port="19530")


def get_or_create_collection(collection_name: str) -> Collection:
    all_collections = m_utility.list_collections()
    if collection_name in all_collections:
        m_collection = Collection(collection_name)
    else:
        # todo: document_id
        fields = [
            FieldSchema(name="id_hash", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=24),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="emb_v1", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="doc_title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=256)
        ]
        schema = CollectionSchema(fields)
        m_collection = Collection(collection_name, schema)
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        m_collection.create_index(
            field_name="emb_v1",
            index_params=index_params
        )
        # todo: do we need an index on primary key? we do if it is not auto, need to check
    m_collection.load()
    return m_collection
