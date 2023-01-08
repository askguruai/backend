import hashlib
import os
import random
import os.path as osp
import json
from typing import List
import shutil


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
            return data
        return None

    def __setitem__(self, hash_: str, data: List):
        fpath = osp.join(self.storage_root, f"{hash_}.json")
        with open(fpath, "wt") as f:
            json.dump(data, f)

    def __delitem__(self, hash_):
        fpath = osp.join(self.storage_root, f"{hash_}.json")
        os.remove(fpath)
