import hashlib
import json
import os
import os.path as osp
import random
from copy import deepcopy
from typing import List

import numpy as np


class StorageDispatcher:
    def __init__(self, storage_root):
        self.storage_root = storage_root

    @staticmethod
    def get_hash(content: str = None) -> str:
        if content is None:
            return str(hex(random.getrandbits(256)))[2:]
        else:
            return hashlib.sha256(content.encode()).hexdigest()

    def __contains__(self, hash_) -> bool:
        fpath = osp.join(self.storage_root, f"{hash_}.json")
        return osp.exists(fpath)

    def __getitem__(self, hash_: str) -> [List, None]:
        fpath = osp.join(self.storage_root, f"{hash_}.json")
        if osp.exists(fpath):
            with open(fpath, "rt") as f:
                data = json.load(f)
            for item in data:
                item["embedding"] = np.array(item["embedding"])
            return data
        return None

    def __setitem__(self, hash_: str, data: List):
        fpath = osp.join(self.storage_root, f"{hash_}.json")
        if len(data) == 0:
            return
        if isinstance(data[0]["embedding"], np.ndarray):
            loc_data = deepcopy(data)
            for item in loc_data:
                item["embedding"] = item["embedding"].tolist()
        else:
            loc_data = data
        with open(fpath, "wt") as f:
            json.dump(loc_data, f)

    def __delitem__(self, hash_):
        fpath = osp.join(self.storage_root, f"{hash_}.json")
        os.remove(fpath)
