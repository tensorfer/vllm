# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from abc import abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import is_non_overlapping_and_dense

logger = logging.getLogger(__name__)


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


def build_mem_zones(kv_caches: dict[str, torch.Tensor]) -> list[tuple[int, int]]:
    mem_zones: list[tuple[int, int]] = []
    seen_zones_ptrs: set[int] = set()

    for cache in kv_caches.values():
        zone = cache.untyped_storage()
        zone_addr = zone.data_ptr()
        zone_bytes = zone.nbytes()

        if zone_addr not in seen_zones_ptrs:
            seen_zones_ptrs.add(zone_addr)
            mem_zones.append((zone_addr, zone_bytes))

    return mem_zones


def build_mem_tensors(kv_caches: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    mem_tensors: list[torch.Tensor] = []
    seen_tensors_ptrs: set[int] = set()

    for layer_name, kv_tensor in kv_caches.items():
        registered = False

        # FA:other = 1:N, so only register FA layer
        if "self_attn" in layer_name and kv_tensor.data_ptr() not in seen_tensors_ptrs:
            seen_tensors_ptrs.add(kv_tensor.data_ptr())
            mem_tensors.append(kv_tensor)
            registered = True

        logger.debug(
            "register_kvcache: %s [%s], address=(%d, %d), "
            "block_bytes=%d, shape=%s, dtype=%s",
            "registered" if registered else "skipped",
            layer_name,
            kv_tensor.data_ptr(),
            kv_tensor.nbytes,
            kv_tensor.stride(0) * kv_tensor.element_size(),
            kv_tensor.shape,
            kv_tensor.dtype,
        )

    return mem_tensors


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
class MemRegion:
    offset: int
    tensors: list[torch.Tensor]


def get_mem_regions(
    kv_caches: list[torch.Tensor],
    block_ids: list[list[int]],
) -> list[MemRegion]:
    """Build file-contiguous MemRegions spanning all caches.

    Adjacent non-placeholder block groups — even across cache boundaries —
    share one region so each contiguous file range is a single I/O.
    """
    grouped = group_block_contiguous([b for g in block_ids for b in g])

    regions: list[MemRegion] = []
    current: MemRegion | None = None
    file_offset = 0

    for cache in kv_caches:
        if not is_non_overlapping_and_dense(cache[0]):
            raise ValueError("Not support `*H*B*` layout for KV cache")
        block_bytes = cache.stride(0) * cache.element_size()
        num_blocks = cache.shape[0]

        for group in grouped:
            group_bytes = len(group) * block_bytes
            if group[0] == 0:
                if current is not None:
                    regions.append(current)
                    current = None
            else:
                if group[0] + len(group) > num_blocks:
                    raise ValueError(
                        f"block range [{group[0]}, {group[0] + len(group)}) "
                        f"out of bound for cache with {num_blocks} blocks"
                    )
                if current is None:
                    current = MemRegion(offset=file_offset, tensors=[])
                current.tensors.append(cache[group[0] : group[0] + len(group)])
            file_offset += group_bytes

    if current is not None:
        regions.append(current)
    return regions


def copy_data_h2d(
    host_data: torch.Tensor,
    dev_data_list: list[torch.Tensor],
    dev_index: int = 0,
    dev_off: int = 0,
    host_bytes: int = 0,
    copy: bool = True,
) -> tuple[int, int]:
    """Copy at most ``host_bytes`` of ``host_data`` into ``dev_data_list``.

    Each device tensor is written in physical (storage) order so the host
    buffer byte layout matches a DIRECT/SGE transfer using the tensor's
    ``data_ptr()``/``nbytes``. This lets save/load bounce and direct policies
    be mixed without reordering permuted (e.g. LBHNC) KV caches.

    Args:
        host_data: Flat source tensor on host.
        dev_data_list: Destination tensors, filled sequentially in physical
            storage order.
        dev_index: Index of the destination tensor to start copying into.
        dev_off: Element offset within ``dev_data_list[dev_index]``'s physical
            storage.
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
            d = dev_data_list[idx]
            dp = d.as_strided((d.numel(),), (1,))
            if off >= dp.numel():
                idx += 1
                off = 0
                continue
            n = min(dp.numel() - off, host_numel - host_off)
            if copy:
                dp[off : off + n].copy_(
                    host_data[host_off : host_off + n], non_blocking=True
                )
            host_off += n
            off += n
            if off == dp.numel():
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

    Each device tensor is read in physical (storage) order so the host buffer
    byte layout matches a DIRECT/SGE transfer using the tensor's
    ``data_ptr()``/``nbytes``. This lets save/load bounce and direct policies
    be mixed without reordering permuted (e.g. LBHNC) KV caches.

    Args:
        host_data: Flat destination tensor on host.
        dev_data_list: Source tensors, consumed sequentially in physical
            storage order.
        dev_index: Index of the source tensor to start copying from.
        dev_off: Element offset within ``dev_data_list[dev_index]``'s physical
            storage.
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
            d = dev_data_list[idx]
            dp = d.as_strided((d.numel(),), (1,))
            if off >= dp.numel():
                idx += 1
                off = 0
                continue
            n = min(dp.numel() - off, host_numel - host_off)
            if copy:
                host_data[host_off : host_off + n].copy_(
                    dp[off : off + n], non_blocking=True
                )
            host_off += n
            off += n
            if off == dp.numel():
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
