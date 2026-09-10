# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from dataclasses import dataclass

import torch

from vllm.utils.torch_utils import is_non_overlapping_and_dense


@dataclass
class ExOffloadingStorageKVCacheConfig:
    kv_caches: dict[str, torch.Tensor]
    num_blocks: int


class ExOffloadingStorage:
    def __init__(self, config):
        self.config = config

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[dict[str, str], str]:
        raise NotImplementedError

    @abstractmethod
    def register_kvcache(self, config: ExOffloadingStorageKVCacheConfig) -> None: ...

    @abstractmethod
    async def load(
        self, filepath: str, offset: int, block_ids: list[list[int]]
    ) -> None: ...

    @abstractmethod
    async def save(
        self, filepath: str, offset: int, block_ids: list[list[int]]
    ) -> None: ...


def build_mem_zones(
    kvcache_config: ExOffloadingStorageKVCacheConfig,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    mem_zones: list[tuple[int, int]] = []
    mem_region_addrs: list[tuple[int, int]] = []
    mem_region_block_bytes: list[int] = []

    seen_zones_ptrs: set[int] = set()
    seen_region_ptrs: dict[int, tuple[int, int]] = {}

    num_blocks = kvcache_config.num_blocks

    for cache in kvcache_config.kv_caches.values():
        zone = cache.untyped_storage()
        zone_addr = zone.data_ptr()
        zone_bytes = zone.nbytes()

        if zone_addr not in seen_zones_ptrs:
            seen_zones_ptrs.add(zone_addr)
            mem_zones.append((zone_addr, zone_bytes))

        if not is_non_overlapping_and_dense(cache[0]):
            raise ValueError("Not support `*H*B*` layout for KV cache")

        block_bytes = cache.stride(0) * cache.element_size()

        if block_bytes * num_blocks == zone_bytes:
            addr = zone_addr
            nbytes = zone_bytes
        else:
            addr = cache.data_ptr()
            nbytes = cache.nbytes

        if addr not in seen_region_ptrs or addr > seen_region_ptrs[addr][0]:
            seen_region_ptrs[addr] = (nbytes, block_bytes)

    for addr, (nbytes, block_bytes) in seen_region_ptrs.items():
        mem_region_addrs.append((addr, nbytes))
        mem_region_block_bytes.append(block_bytes)

    return mem_zones, mem_region_addrs, mem_region_block_bytes


def group_block_contiguous(block_ids: list[int]) -> list[list[int]]:
    if not block_ids:
        return []

    groups: list[list[int]] = []
    group_start = 0
    for index in range(1, len(block_ids)):
        previous_id = block_ids[index - 1]
        current_id = block_ids[index]
        is_contiguous = current_id in (previous_id, previous_id + 1)
        is_placeholder_transition = previous_id == 0 and current_id != 0
        if not is_contiguous or is_placeholder_transition:
            groups.append(block_ids[group_start:index])
            group_start = index

    groups.append(block_ids[group_start:])
    return groups


@dataclass
class RegionDesc:
    offset: int
    address: list[tuple[int, int]]


def _build_mem_regions_for_zone(
    region_addr: int,
    region_size: int,
    block_bytes: int,
    grouped_block_ids: list[list[int]],
    file_offset: int,
) -> tuple[list[RegionDesc], int]:
    """Build file-to-memory mappings for one memory zone."""
    regions: list[RegionDesc] = []
    current_region: RegionDesc | None = None

    for group in grouped_block_ids:
        group_size = len(group) * block_bytes

        if group[0] == 0:
            if current_region is not None:
                regions.append(current_region)
                current_region = None
            file_offset += group_size
            continue

        group_addr = region_addr + group[0] * block_bytes
        group_end = group_addr + group_size
        region_end = region_addr + region_size
        if group_end > region_end:
            raise ValueError(
                f"memory region [{group_addr}, {group_end}] is out of bound"
            )

        if current_region is None:
            current_region = RegionDesc(offset=file_offset, address=[])

        current_region.address.append((group_addr, group_size))
        file_offset += group_size

    if current_region is not None:
        regions.append(current_region)

    return regions, file_offset


def get_mem_regions(
    mem_region_addrs: list[tuple[int, int]],
    mem_region_block_bytes: list[int],
    block_ids: list[list[int]],
) -> list[RegionDesc]:
    """Map logical block IDs to file offsets and GPU memory addresses."""
    if len(mem_region_addrs) != len(mem_region_block_bytes):
        raise ValueError(
            "mem_region_addrs and mem_region_block_bytes must have the same length"
        )

    flattened_block_ids = [block_id for group in block_ids for block_id in group]
    grouped_block_ids = group_block_contiguous(flattened_block_ids)

    mem_regions: list[RegionDesc] = []
    file_offset = 0

    for (region_addr, region_size), block_bytes in zip(
        mem_region_addrs, mem_region_block_bytes
    ):
        zone_regions, file_offset = _build_mem_regions_for_zone(
            region_addr,
            region_size,
            block_bytes,
            grouped_block_ids,
            file_offset,
        )
        mem_regions.extend(zone_regions)

    return mem_regions
