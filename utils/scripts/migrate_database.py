import os
import sys
from collections import defaultdict

sys.path.insert(1, os.getcwd())

from pymilvus import Collection, utility

from utils import MILVUS_DB, hash_string
from utils.milvus_utils import MilvusSchema

OLD_DB_NAME = "livechat_77b02fc338f14068af94440b_chats"
# OLD_DB_NAME = None

VENDOR = "askgurupublic"
ORGANIZATION = "askgurupublic"
COLLECTION = "test2"


OLD_SCHEMA_FIELDS = [
    "pk",
    "chunk_hash",
    "doc_id",
    "chunk",
    "emb_v1",
    "doc_title",
    "doc_summary",
    "timestamp",
    "security_groups",
]
NEW_SCHEMA_FIELDS = [
    "pk",
    "chunk_hash",
    "doc_id",
    "chunk",
    "emb_v1",
    "doc_title",
    "doc_summary",
    "timestamp",
    "security_groups",
    "url",
]


def assert_schema(collection: Collection, expected_fields):
    schema = collection.schema
    field_names = {field.name for field in schema.fields}
    assert field_names == set(expected_fields), field_names
    print(f"Schema assured for {collection.name}")


if OLD_DB_NAME is not None:
    old_db_name = OLD_DB_NAME
else:
    old_db_name = f"{VENDOR}_{hash_string(ORGANIZATION)}_{COLLECTION}"
assert old_db_name in utility.list_collections()
old_db = MILVUS_DB[old_db_name]
old_db.load()
assert_schema(old_db, OLD_SCHEMA_FIELDS)
# old_db_entities = old_db.num_entities
# print(f"Entites in old_db: {old_db_entities}")
data = old_db.query(
    expr=f"pk>=0",
    offset=0,
    limit=5000,
    output_fields=OLD_SCHEMA_FIELDS,
    consistency_level="Strong",
)
queried = len(data)
print(f"Queried from old collection: {queried}")
# assert old_db_entities == queried, "Exiting"


new_db_name = f"new_{old_db_name}"
if new_db_name in utility.list_collections():
    utility.drop_collection(new_db_name)
new_db = MILVUS_DB.get_or_create_collection(new_db_name, schema=MilvusSchema.V1)

datas = defaultdict(list)
for hit in data:
    for field in OLD_SCHEMA_FIELDS:
        datas[field].append(hit[field])
insert_data = [datas[field] for field in OLD_SCHEMA_FIELDS[1:]]  # skipping pk field
insert_data.append([""] * len(datas["chunk_hash"]))  # new url field set to empty
new_db.insert(insert_data)

utility.flush_all()
new_db_ent = new_db.num_entities
assert new_db_ent == queried, f"Old count (queried): {queried}, new count: {new_db_ent}"

utility.rename_collection(old_db_name, f"temp_{old_db_name}")
utility.rename_collection(new_db_name, old_db_name)
assert_schema(new_db, NEW_SCHEMA_FIELDS)

print(f"All good!")
