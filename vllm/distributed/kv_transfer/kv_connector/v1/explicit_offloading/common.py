# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorageKVCacheConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.manager import (  # noqa: E501
    ExOffloadingStorageManager,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

logger = logging.getLogger(__name__)


@dataclass
class ExKVCacheSegment:
    """
    Represents a segment of cached tokens and their corresponding KV cache location.

    Attributes:
        token_start: The starting token ID within the logical token sequence.
        token_length: The number of tokens in this segment.
        kv_uri: A URI string identifying the storage backend and path for the KV cache.
        kv_start: The starting position in the physical KV cache storage.
        kv_length: The length of the KV cache data in the storage (optional).
    """

    token_start: int
    token_length: int
    block_size: int
    kv_uri: str
    kv_start: int
    kv_length: int | None = None

    def __post_init__(self):
        if self.token_start < 0:
            raise ValueError(
                f"token_start must be non-negative, got {self.token_start}"
            )
        if self.token_length <= 0 and self.token_length != -1:
            raise ValueError(
                f"token_length must be positive or -1(for infinite), "
                f"got {self.token_length}"
            )
        if self.kv_start < 0:
            raise ValueError(f"kv_start must be non-negative, got {self.kv_start}")
        if self.kv_length is not None and self.kv_length <= 0:
            raise ValueError(f"kv_length must be positive if set, got {self.kv_length}")

        if self.token_start % self.block_size != 0:
            raise ValueError(
                f"token_start ({self.token_start}) must be aligned to "
                f"block_size ({self.block_size})"
            )

        if self.token_length == -1:
            self.token_length = (sys.maxsize // self.block_size) * self.block_size
        if self.token_length % self.block_size != 0:
            raise ValueError(
                f"token_length ({self.token_length}) must be aligned to "
                f"block_size ({self.block_size})"
            )

    @classmethod
    def from_dict(cls, d: dict, block_size: int) -> "ExKVCacheSegment":
        """Creates an instance from a dictionary."""
        required_fields = ["token_start", "token_length", "kv_uri", "kv_start"]
        missing_fields = [f for f in required_fields if f not in d]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in ExKVCacheSegment dictionary: "
                f"{missing_fields}. Required fields: {required_fields}"
            )
        return cls(
            token_start=d["token_start"],
            token_length=d["token_length"],
            block_size=block_size,
            kv_uri=d["kv_uri"],
            kv_start=d["kv_start"],
            kv_length=d.get("kv_length"),
        )

    def to_dict(self) -> dict:
        """Converts the instance to a dictionary."""
        return {
            "token_start": self.token_start,
            "token_length": self.token_length,
            "kv_uri": self.kv_uri,
            "kv_start": self.kv_start,
            "kv_length": self.kv_length,
        }

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_length

    @property
    def block_start(self) -> int:
        return self.token_start // self.block_size

    @property
    def block_length(self) -> int:
        return self.token_length // self.block_size

    @property
    def block_end(self) -> int:
        return self.token_end // self.block_size


class ExKVCacheContext:
    def __init__(
        self,
        segments: list[ExKVCacheSegment] | list[dict[str, Any]] | None = None,
        block_size: int = 0,
        block_ids: tuple[list[int], ...] | None = None,
        kv_cache_groups: list[KVCacheGroupSpec] | None = None,
        kv_length_per_block: int | None = None,
    ):
        # block size for token
        self._block_size: int = block_size
        self._block_ids: tuple[list[int], ...] | None = None
        self._kv_cache_groups: list[KVCacheGroupSpec] | None = None
        # size of each physical block page slot in bytes
        self._kv_length_per_block: int | None = kv_length_per_block
        self._offset: int = 0

        if not segments:
            self._segments: tuple[ExKVCacheSegment, ...] = ()
            return

        processed: list[ExKVCacheSegment] = []
        for seg in segments:
            if isinstance(seg, dict):
                processed.append(
                    ExKVCacheSegment.from_dict(seg, block_size=self._block_size)
                )
            else:
                processed.append(seg)

        if not processed:
            self._segments = ()
            return

        self._segments = tuple(sorted(processed, key=lambda s: s.token_start))

        self._check_segments()

        self.bind_block_ids(block_ids, kv_cache_groups)

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, index: int) -> ExKVCacheSegment:
        return self._segments[index]

    def __iter__(self) -> Iterator[ExKVCacheSegment]:
        return iter(self._segments)

    def __repr__(self) -> str:
        return f"ExKVCacheContext(segments={list(self._segments)})"

    def _check_segments(self):
        prev_end = self._segments[0].token_start
        for i, seg in enumerate(self._segments):
            if seg.token_start != prev_end:
                raise ValueError(
                    f"Segments are not contiguous at index {i}: "
                    f"...[{self._segments[i - 1].token_start}, "
                    f"{self._segments[i - 1].token_end}], "
                    f"[{seg.token_start}, {seg.token_end}]..."
                    if i > 0
                    else ""
                )
            prev_end = seg.token_end

    @property
    def token_start(self) -> int:
        """The logical start token ID of the entire cache."""
        return self._segments[0].token_start + self._offset if self._segments else 0

    @property
    def token_end(self) -> int:
        """The logical end token ID (exclusive) of the entire cache."""
        return self._segments[-1].token_end if self._segments else 0

    @property
    def token_length(self) -> int:
        """The total logical token length covered by the cache."""
        return self.token_end - self.token_start

    @property
    def token_range(self) -> tuple[int, int] | None:
        """Returns the (start, end) logical token range, or None if empty."""
        if not self._segments:
            return None
        return (self.token_start, self.token_end)

    @property
    def real_token_start(self) -> int:
        """The real start token ID of the entire cache."""
        return self._segments[0].token_start if self._segments else 0

    @property
    def real_token_end(self) -> int:
        """The real end token ID (exclusive) of the entire cache."""
        return self.token_end

    @property
    def real_token_length(self) -> int:
        """The real token length including the internal offset."""
        return self.real_token_end - self.real_token_start

    @property
    def block_offset(self) -> int:
        """The block offset corresponding to the internal token offset."""
        return self._offset // self._block_size

    @property
    def block_start(self) -> int:
        """The starting block index of the entire cache."""
        return (
            self._segments[0].block_start + self.block_offset if self._segments else 0
        )

    @property
    def block_end(self) -> int:
        """The ending block index of the entire cache."""
        return self._segments[-1].block_end if self._segments else 0

    @property
    def block_length(self) -> int:
        """The total block length covered by the cache."""
        return self.block_end - self.block_start

    @property
    def block_range(self) -> tuple[int, int] | None:
        """Returns the (start, end) block range, or None if empty."""
        if not self._segments:
            return None
        return (self.block_start, self.block_end)

    @property
    def real_block_start(self) -> int:
        """The real starting block index of the entire cache."""
        return self._segments[0].block_start if self._segments else 0

    @property
    def real_block_end(self) -> int:
        """The real ending block index of the entire cache."""
        return self.block_end

    @property
    def real_block_length(self) -> int:
        """The real block length including the internal offset."""
        return self.real_block_end - self.real_block_start

    @property
    def block_ids(self) -> list[list[int]] | None:
        if not self._block_ids or not self._kv_cache_groups:
            return None

        block_ids: list[list[int]] = []
        for seg in self._segments:
            seg_block_ids = self.get_block_ids(seg)
            assert seg_block_ids is not None

            block_ids = [
                group_block_ids + group_seg_block_ids
                for group_block_ids, group_seg_block_ids in zip(
                    block_ids, seg_block_ids
                )
            ]

        return block_ids

    @property
    def kv_length(self) -> int | None:
        length: int = 0
        for seg in self._segments:
            if seg.kv_length is None:
                return None
            length += seg.kv_length
        return length

    def result(self, tp_size: int = 1, use_mla: bool = False) -> list[dict]:
        if self._kv_length_per_block is None:
            raise ValueError("Missing kv_length_per_block")

        if use_mla:
            tp_size = 1

        result = []
        for seg in self._segments:
            block_count = self.get_block_count(seg)
            assert block_count > 0
            new_kv_length = block_count * self._kv_length_per_block * tp_size

            result.append(
                ExKVCacheSegment(
                    token_start=seg.token_start,
                    token_length=seg.token_length,
                    block_size=self._block_size,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=new_kv_length,
                ).to_dict()
            )

        return result

    def bind_block_ids(
        self,
        block_ids: tuple[list[int], ...] | None,
        kv_cache_groups: list[KVCacheGroupSpec] | None,
    ) -> "ExKVCacheContext":
        if not self._segments:
            return self

        if block_ids is None and kv_cache_groups is None:
            return self

        if block_ids is None or kv_cache_groups is None:
            raise ValueError(
                "bind_block_ids requires both block_ids "
                "and kv_cache_groups to be provided"
            )

        if len(block_ids) != len(kv_cache_groups):
            raise ValueError("Length of block_ids must match length of kv_cache_groups")

        self._block_ids = block_ids
        self._kv_cache_groups = kv_cache_groups

        return self

    def get_block_ids(self, seg: ExKVCacheSegment) -> list[list[int]] | None:
        if not self._block_ids or not self._kv_cache_groups:
            return None

        is_last_seg = seg == self._segments[-1]

        block_ids: list[list[int]] = []
        for group_block_ids, group_spec in zip(self._block_ids, self._kv_cache_groups):
            if isinstance(group_spec.kv_cache_spec, FullAttentionSpec):
                block_ids.append(group_block_ids[seg.block_start : seg.block_end])
            elif isinstance(group_spec.kv_cache_spec, SlidingWindowSpec):
                sw_size = group_spec.kv_cache_spec.sliding_window
                block_size = group_spec.kv_cache_spec.block_size
                blocks_per_sw = sw_size // block_size
                if is_last_seg:
                    # Clip the block IDs to the sliding window size.
                    start = max(seg.block_start, seg.block_end - blocks_per_sw)
                    cliped_block_ids = group_block_ids[start : seg.block_end]
                    assert all(block_id != 0 for block_id in cliped_block_ids)
                    block_ids.append(cliped_block_ids)
                else:
                    # Placeholder block IDs for non-last segments in SW layer
                    block_ids.append([0] * blocks_per_sw)
            elif isinstance(group_spec.kv_cache_spec, MambaSpec):
                if is_last_seg:
                    end_block_id = group_block_ids[seg.block_end - 1]
                    assert end_block_id != 0
                    block_ids.append([end_block_id])
                else:
                    # Placeholder block IDs for non-last segments in Mamba layer
                    block_ids.append([0])
            else:
                raise ValueError(
                    f"Unsupported KVCacheSpec type: {type(group_spec.kv_cache_spec)}"
                )

        return block_ids

    def get_block_count(self, seg: ExKVCacheSegment) -> int:
        group_block_ids = self.get_block_ids(seg)
        if group_block_ids is None:
            return -1

        return sum(len(ids) for ids in group_block_ids)

    def reset(self) -> "ExKVCacheContext":
        self._segments = ()
        self._offset = 0
        self._block_ids = None
        self._kv_cache_groups = None
        return self

    def update_kv_layout(
        self,
        kv_length_per_block: int | None = None,
        tp_rank: int | None = None,
        replicates_kv_cache: bool = False,
    ) -> "ExKVCacheContext":
        if kv_length_per_block is not None:
            if self._kv_length_per_block is not None:
                raise ValueError("kv_length_per_block is already set")
            self._kv_length_per_block = kv_length_per_block

        if tp_rank is not None:
            if self._kv_length_per_block is None:
                raise ValueError("Setting tp_rank requires kv_length_per_block")

            new_segments = []
            for seg in self._segments:
                block_count = self.get_block_count(seg)
                assert block_count > 0
                kv_length = block_count * self._kv_length_per_block

                kv_start = seg.kv_start
                if not replicates_kv_cache:
                    kv_start = seg.kv_start + tp_rank * kv_length

                new_segments.append(
                    ExKVCacheSegment(
                        token_start=seg.token_start,
                        token_length=seg.token_length,
                        block_size=seg.block_size,
                        kv_uri=seg.kv_uri,
                        kv_start=kv_start,
                        kv_length=kv_length,
                    )
                )

            self._segments = tuple(new_segments)
            self._check_segments()

        return self

    def truncate_prefix(self, block_offset: int) -> "ExKVCacheContext":
        if not self._segments:
            return self

        if block_offset < self.block_start:
            raise ValueError(
                f"block_offset ({block_offset}) must be >= cache start "
                f"({self.block_start})"
            )

        if block_offset >= self.block_end:
            return self.reset()

        new_segments = [seg for seg in self._segments if seg.block_end > block_offset]
        if not new_segments:
            return self.reset()

        self._offset = (block_offset - new_segments[0].block_start) * self._block_size
        self._segments = tuple(new_segments)
        self._check_segments()

        return self

    def truncate_suffix(self, block_offset: int) -> "ExKVCacheContext":
        if not self._segments:
            return self

        if block_offset <= self.block_start:
            return self.reset()

        if block_offset >= self.block_end:
            return self

        new_segments = []
        for seg in self._segments:
            if seg.block_start >= block_offset:
                break
            elif seg.block_end <= block_offset:
                new_segments.append(seg)
            else:
                new_token_length = (block_offset - seg.block_start) * seg.block_size

                new_kv_length = None
                block_count = self.get_block_count(seg)
                if block_count > 0 and self._kv_length_per_block is not None:
                    new_kv_length = block_count * self._kv_length_per_block

                new_seg = ExKVCacheSegment(
                    token_start=seg.token_start,
                    token_length=new_token_length,
                    block_size=seg.block_size,
                    kv_uri=seg.kv_uri,
                    kv_start=seg.kv_start,
                    kv_length=new_kv_length,
                )
                new_segments.append(new_seg)
                break

        if not new_segments:
            return self.reset()

        self._segments = tuple(new_segments)
        self._check_segments()

        return self

    async def _prefetch_seg(
        self, seg: ExKVCacheSegment, kvcache_config: ExOffloadingStorageKVCacheConfig
    ):
        block_ids = self.get_block_ids(seg)
        assert block_ids is not None

        storage, path = ExOffloadingStorageManager.get_storage_by_uri(
            seg.kv_uri, kvcache_config
        )

        await storage.load(path, seg.kv_start, block_ids)

    async def prefetch(self, kvcache_config: ExOffloadingStorageKVCacheConfig):
        if self._block_ids is None:
            raise ValueError("block_ids are not bound")

        t0 = time.perf_counter()

        await asyncio.gather(
            *[self._prefetch_seg(seg, kvcache_config) for seg in self._segments]
        )

        if logger.isEnabledFor(logging.DEBUG):
            elapsed = time.perf_counter() - t0
            kv_len = self.kv_length or 0
            bps = kv_len / elapsed if elapsed > 0 else float("inf")
            logger.debug(
                "EXKV prefetch %s: %s in %.3fs (%s/s)",
                self,
                bytes_to_human(kv_len),
                elapsed,
                bytes_to_human(bps),
            )

    async def backup(self, kvcache_config: ExOffloadingStorageKVCacheConfig):
        t0 = time.perf_counter()

        for seg in self._segments:
            block_ids = self.get_block_ids(seg)
            assert block_ids is not None

            storage, path = ExOffloadingStorageManager.get_storage_by_uri(
                seg.kv_uri, kvcache_config
            )

            await storage.save(path, seg.kv_start, block_ids)

        if logger.isEnabledFor(logging.DEBUG):
            elapsed = time.perf_counter() - t0
            kv_len = self.kv_length or 0
            bps = kv_len / elapsed if elapsed > 0 else float("inf")
            logger.debug(
                "EXKV backup %s: %s in %.3fs (%s/s)",
                self,
                bytes_to_human(kv_len),
                elapsed,
                bytes_to_human(bps),
            )


@dataclass
class ExOffloadingRequestContext:
    id: str
    request_id: str
    exkvcache: ExKVCacheContext


@dataclass
class ExOffloadingConnectorMetadata(KVConnectorMetadata):
    load_req_ctx: list[ExOffloadingRequestContext]
    save_req_ctx: list[ExOffloadingRequestContext]


def bytes_to_human(n: float, precision: int = 2) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB")
    sign = "-" if n < 0 else ""
    size = abs(n)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{sign}{int(size)} {unit}"
            return f"{sign}{size:.{precision}f} {unit}"
        size /= 1024

    return ""
