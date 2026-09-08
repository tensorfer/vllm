# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm.utils.math_utils import round_up
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


@dataclass
class RegionTensor:
    offset: int
    tensors: list[torch.Tensor]


def _build_mem_tensors_for_cache(
    cache: torch.Tensor,
    block_bytes: int,
    grouped_block_ids: list[list[int]],
    file_offset: int,
) -> tuple[list[RegionTensor], int]:
    tensors: list[RegionTensor] = []
    current_region: RegionTensor | None = None
    num_blocks = cache.shape[0]

    for group in grouped_block_ids:
        group_size = len(group) * block_bytes

        if group[0] == 0:
            if current_region is not None:
                tensors.append(current_region)
                current_region = None
            file_offset += group_size
            continue

        if group[0] + len(group) > num_blocks:
            raise ValueError(
                f"block range [{group[0]}, {group[0] + len(group)}) "
                f"out of bound for cache with {num_blocks} blocks"
            )

        if current_region is None:
            current_region = RegionTensor(offset=file_offset, tensors=[])

        current_region.tensors.append(cache[group[0] : group[0] + len(group)])
        file_offset += group_size

    if current_region is not None:
        tensors.append(current_region)

    return tensors, file_offset


def get_mem_tensors(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
) -> list[RegionTensor]:
    flattened_block_ids = [block_id for group in block_ids for block_id in group]
    grouped_block_ids = group_block_contiguous(flattened_block_ids)

    mem_tensors: list[RegionTensor] = []
    file_offset = 0

    for cache in kv_caches.values():
        if not is_non_overlapping_and_dense(cache[0]):
            raise ValueError("Not support `*H*B*` layout for KV cache")

        block_bytes = cache.stride(0) * cache.element_size()
        cache_tensors, file_offset = _build_mem_tensors_for_cache(
            cache,
            block_bytes,
            grouped_block_ids,
            file_offset,
        )
        mem_tensors.extend(cache_tensors)

    return mem_tensors


def copy_data_h2d(
    host_data: torch.Tensor,
    dev_data_list: list[torch.Tensor],
    dev_index: int = 0,
    dev_off: int = 0,
    host_bytes: int = 0,
    copy: bool = True,
) -> tuple[int, int]:
    """Copy at most ``host_bytes`` of ``host_data`` into ``dev_data_list``.
    Args:
        host_data: Flat source tensor on host.
        dev_data_list: Destination tensors, filled sequentially once flattened.
        dev_index: Index of the destination tensor to start copying into.
        dev_off: Element offset within ``dev_data_list[dev_index]``.
        host_bytes: Number of valid bytes in ``host_data``; 0 means all of it.
        copy: When False, only compute the next destination position.
    Returns:
        The ``dev_index``/``dev_off`` where the next copy round should resume.
    """
    if host_bytes == 0:
        host_bytes = host_data.element_size() * host_data.numel()
    host_numel = host_bytes // host_data.element_size()

    if not dev_data_list:
        return dev_index, dev_off

    use_cuda = dev_data_list[0].is_cuda
    stream = torch.cuda.Stream() if use_cuda and copy else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        idx, off = dev_index, dev_off
        host_off = 0
        while host_off < host_numel and idx < len(dev_data_list):
            df = dev_data_list[idx].flatten()
            if off >= df.numel():
                idx += 1
                off = 0
                continue
            n = min(df.numel() - off, host_numel - host_off)
            if copy:
                df[off : off + n].copy_(
                    host_data[host_off : host_off + n], non_blocking=True
                )
            host_off += n
            off += n
            if off == df.numel():
                idx += 1
                off = 0

    if stream is not None:
        stream.synchronize()

    return idx, off


def copy_data_d2h(
    host_data: torch.Tensor,
    dev_data_list: list[torch.Tensor],
    dev_index: int = 0,
    dev_off: int = 0,
    host_bytes: int = 0,
    copy: bool = True,
) -> tuple[int, int]:
    """Copy at most ``host_bytes`` of ``dev_data_list`` into ``host_data``.
    Args:
        host_data: Flat destination tensor on host.
        dev_data_list: Source tensors, consumed sequentially once flattened.
        dev_index: Index of the source tensor to start copying from.
        dev_off: Element offset within ``dev_data_list[dev_index]``.
        host_bytes: Number of bytes to write into ``host_data``; 0 means all.
        copy: When False, only compute the next source position.
    Returns:
        The ``dev_index``/``dev_off`` where the next copy round should resume.
    """
    if host_bytes == 0:
        host_bytes = host_data.element_size() * host_data.numel()
    host_numel = host_bytes // host_data.element_size()

    if not dev_data_list:
        return dev_index, dev_off

    use_cuda = dev_data_list[0].is_cuda
    stream = torch.cuda.Stream() if use_cuda and copy else None
    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        idx, off = dev_index, dev_off
        host_off = 0
        while host_off < host_numel and idx < len(dev_data_list):
            df = dev_data_list[idx].flatten()
            if off >= df.numel():
                idx += 1
                off = 0
                continue
            n = min(df.numel() - off, host_numel - host_off)
            if copy:
                host_data[host_off : host_off + n].copy_(
                    df[off : off + n], non_blocking=True
                )
            host_off += n
            off += n
            if off == df.numel():
                idx += 1
                off = 0

    if stream is not None:
        stream.synchronize()

    return idx, off


def copy_data_split_chunks(
    host_data: torch.Tensor,
    dev_data_list: list[torch.Tensor],
    file_off: int,
    file_align: int,
    copyfunc: Callable[..., tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Split a transfer into chunks of at most one ``host_data`` in size.

    Every chunk's file offset is aligned to ``file_align`` except the first
    one, which starts at ``file_off`` and is trimmed to the next alignment
    boundary. ``host_data``'s size is expected to be a multiple of
    ``file_align`` so that only the first chunk may be unaligned.
    Args:
        host_data: Chunk-sized host tensor (a bounce slot); its size in
            bytes bounds every chunk.
        dev_data_list: Device-side tensors of the transfer, walked in order.
        file_off: File offset the data starts at.
        file_align: Alignment expected of every chunk's file offset except
            the first one.
        copyfunc: ``copy_data_h2d`` or ``copy_data_d2h``, invoked with
            ``copy=False`` to walk device positions only.
    Returns:
        Chunks of ``(file_off, length, dev_index, dev_off)`` covering the
        tensors in order.
    """
    if not dev_data_list:
        return []

    assert file_align > 0
    chunk_len = host_data.element_size() * host_data.numel()
    total_bytes = host_data.element_size() * tensors_total_numel(dev_data_list)

    chunks: list[tuple[int, int, int, int]] = []
    dev_index, dev_off = 0, 0
    remaining = total_bytes
    while remaining > 0:
        length = min(chunk_len, remaining)
        if file_off % file_align != 0:
            length = min(length, round_up(file_off, file_align) - file_off)
        chunks.append((file_off, length, dev_index, dev_off))
        dev_index, dev_off = copyfunc(
            host_data,
            dev_data_list,
            dev_index,
            dev_off,
            host_bytes=length,
            copy=False,
        )
        file_off += length
        remaining -= length

    return chunks


def tensors_total_numel(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors)
