import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(1, os.getcwd())

from loguru import logger

from utils import MILVUS_DB

COL_NAME, TOTAL, BATCH = "test_load", 20000, 500


def insert(collection, amount, batch):
    for _ in tqdm(range(0, amount, batch)):
        data = [
            [""] * batch,
            [""] * batch,
            [""] * batch,
            np.random.rand(batch, 1536),
            [""] * batch,
            [""] * batch,
            [0] * batch,
            [2**63 - 1] * batch,
        ]
        collection.insert(data)


def search():
    logger.info("Starting search!")
    _, chunks, titles, doc_ids, doc_summaries, doc_collections = MILVUS_DB.search_collections_set(
        [COL_NAME],
        np.random.rand(1536),
        50,
        "v1",
    )
    logger.info("Search finished!")


def main():
    collection = MILVUS_DB.get_or_create_collection(COL_NAME)
    # insert(collection, TOTAL, BATCH)
    logger.info(f"Num entities in collection: {collection.num_entities}")
    search()


if __name__ == "__main__":
    main()
