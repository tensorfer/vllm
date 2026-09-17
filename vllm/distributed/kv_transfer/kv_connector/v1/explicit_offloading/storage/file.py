# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorage,
    ExOffloadingStorageKVCacheConfig,
)

DEFAULT_FILE_ROOT_PATH = "/dev/shm/vllm_file_storage"


class FileStorage(ExOffloadingStorage):
    def __init__(self, extra_config: dict[str, str]):
        self.root_path = extra_config.get("root_path")
        if self.root_path is None:
            raise ValueError("not found File root root_path")

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[dict[str, str], str]:
        if not uri.startswith("file://"):
            raise ValueError("invalid File URI format")

        filepath = uri.replace("file://", "")

        root_path = os.getenv("FILE_ROOT_PATH", None)
        if root_path is not None:
            if not os.path.exists(root_path):
                raise ValueError(f"FILE_ROOT_PATH {root_path} not exists")
        else:
            root_path = DEFAULT_FILE_ROOT_PATH
            os.makedirs(root_path, exist_ok=True)

        return {"root_path": root_path}, filepath

    def register_kvcache(self, config: ExOffloadingStorageKVCacheConfig) -> None:
        self.kvcache_config = config

    async def load(
        self, filepath: str, offset: int, block_ids: list[list[int]]
    ) -> None:
        raise NotImplementedError

    async def save(
        self, filepath: str, offset: int, block_ids: list[list[int]]
    ) -> None:
        raise NotImplementedError
