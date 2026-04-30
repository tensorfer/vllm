# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import bisect
import os
import urllib.parse
from enum import Enum
from typing import Any

import pygd2fs
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.net_device import (  # noqa: E501
    DeviceInfo,
    find_closest_nics,
    get_current_device_info,
    get_nic_info_dict,
    get_nic_name_dict,
)
from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorage,
    ExOffloadingStorageKVCacheConfig,
    build_mem_regions,
    copy_data_d2h,
    copy_data_h2d,
    get_mem_regions,
    get_mem_tensors,
    tensors_total_numel,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class GD2FSIOPolicy(Enum):
    DIRECT = "direct"
    BOUNCE = "bounce"


class GD2FSDataPlaneKind(Enum):
    TCP = "tcp"
    RDMA = "rdma"


class DPAddrInfo:
    def __init__(self, uri: str):
        if not uri:
            return

        if "://" not in uri:
            raise ValueError("Invalid DPADDR: missing '://'")

        split = urllib.parse.urlparse(uri)
        self.protocol = split.scheme
        self.address = split.netloc
        self.nic_name: str | None = None
        self.distance_level: DeviceInfo.DistanceLevel | None = None
        self.distance: int | None = None

    def __str__(self) -> str:
        return f"{self.protocol}://{self.address}"

    def __repr__(self) -> str:
        return (
            f"{self.protocol}://{self.address} "
            f"(name={self.nic_name},"
            f"distance={self.distance_level.name if self.distance_level else None}"
            f":{self.distance})"
        )

    def update(
        self,
        nic_name: str,
        distance_level: DeviceInfo.DistanceLevel,
        distance: int,
    ):
        self.nic_name = nic_name
        self.distance_level = distance_level
        self.distance = distance


def get_dpaddr_info_list(dpaddrs: str) -> dict[str, DPAddrInfo]:
    result = {}
    for addr in dpaddrs.split(","):
        addr = addr.strip()
        if not addr:
            continue

        dpaddr_info = DPAddrInfo(addr)

        if dpaddr_info.address in result:
            raise ValueError(
                f"DPADDR {dpaddr_info} and DPADDR {result[dpaddr_info.address]} "
                "have the same address and different protocols"
            )

        result[dpaddr_info.address] = dpaddr_info

    return result


def select_dpaddrs(dpaddrs: str, n: int = 0) -> list[DPAddrInfo]:
    result = []

    device_info = get_current_device_info()

    # addr: dpaddr_info
    dpaddr_info_dict = get_dpaddr_info_list(dpaddrs)
    # nic_name: addr_list
    nic_name_dict = get_nic_name_dict(addr_list=[addr for addr in dpaddr_info_dict])
    # nic_name: device_info
    nic_info_dict = get_nic_info_dict(nic_name_list=[name for name in nic_name_dict])

    selected_nic_list = find_closest_nics(device_info, nic_info_dict, n)
    for nic_name, distance_level, distance in selected_nic_list:
        addr_list = nic_name_dict[nic_name]
        if len(addr_list) > 1:
            logger.warning(
                "nic %s is required by DPADDR to use multiple addresses (%s), "
                "but only the first one can be used",
                nic_name,
                addr_list,
            )
        dpaddr = dpaddr_info_dict[addr_list[0]]
        dpaddr.update(nic_name, distance_level, distance)
        result.append(dpaddr)

    logger.info("%s selected dpaddrs: %s", device_info.name, result)

    return result


def get_protocol(dpaddr_info_list: list[DPAddrInfo]) -> str:
    protocols: set[str] = set()
    for dpaddr_info in dpaddr_info_list:
        protocols.add(dpaddr_info.protocol)

    if len(protocols) > 1:
        raise ValueError(
            f"DPADDR list is required the same protocal, but got {protocols}"
        )

    return protocols.pop()


class GD2FSStorage(ExOffloadingStorage):
    def __init__(self, config: dict[str, Any]):
        self.cpaddr = config.get("cpaddr")
        if self.cpaddr is None:
            raise ValueError("not found GD2FS cpaddr")

        dpaddrs = config.get("dpaddr")
        if dpaddrs is None:
            raise ValueError("not found GD2FS dpaddr")
        nic_count = config.get("nics", 0)
        self.dpaddrs = select_dpaddrs(dpaddrs, n=nic_count)

        dpkind = get_protocol(self.dpaddrs)
        self.dpkind = GD2FSDataPlaneKind(dpkind)

        self.cluster = config.get("cluster")
        if self.cluster is None:
            raise ValueError("not found GD2FS cluster")

        self.iothreads = config.get("iothreads", 1)
        self.memthreads = config.get("memthreads", 1)
        self.streams = config.get("streams", 1)

        self.client = pygd2fs.Client(
            self.cpaddr,
            ",".join([str(dpaddr) for dpaddr in self.dpaddrs]),
            self.cluster,
            iothreads=self.iothreads,
            memthreads=self.memthreads,
            streams=self.streams,
        )
        if self.client is None:
            raise RuntimeError("cannot connect to GD2FS cluster")

        self.mem_regions_per_layer: list[tuple[int, int]] = []
        self.mem_regions_iomem: list[tuple[int, int, pygd2fs.IOMEM]] = []

        self.load_policy = self.get_io_policy("GD2FS_LOAD_POLICY")
        self.save_policy = self.get_io_policy("GD2FS_SAVE_POLICY")
        self.check_io_policy()

        self._load_fn = (
            self._load_direct
            if self.load_policy == GD2FSIOPolicy.DIRECT
            else self._load_bounce
        )

        self._save_fn = (
            self._save_direct
            if self.save_policy == GD2FSIOPolicy.DIRECT
            else self._save_bounce
        )

    def get_io_policy(self, envkey: str) -> GD2FSIOPolicy:
        policy = os.getenv(envkey, None)
        if policy is None:
            if self.dpkind == GD2FSDataPlaneKind.TCP:
                policy = "bounce"
            elif self.dpkind == GD2FSDataPlaneKind.RDMA:
                policy = "direct"
            else:
                raise ValueError(
                    f"GD2FS data plane kind {self.dpkind} is not supported"
                )
        return GD2FSIOPolicy(policy.lower())

    def check_io_policy(self):
        if self.dpkind == GD2FSDataPlaneKind.TCP:
            if self.load_policy == GD2FSIOPolicy.DIRECT:
                raise ValueError(
                    "GD2FS load policy direct is not supported for tcp data plane"
                )

            if self.save_policy == GD2FSIOPolicy.DIRECT:
                raise ValueError(
                    "GD2FS save policy direct is not supported for tcp data plane"
                )

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[dict, str]:
        if not uri.startswith("gd2fs://"):
            raise ValueError("Invalid GD2FS URI")

        parsed = urllib.parse.urlparse(uri)

        addr_list = [a.strip() for a in parsed.netloc.split(",") if a.strip()]
        cpaddr = ",".join(f"gd2fs://{a}" for a in addr_list)

        dpaddr = os.getenv("GD2FS_DPADDR", "tcp://127.0.0.1")
        nics = int(os.getenv("GD2FS_DEVICE_NICS", 0))
        iothreads = int(os.getenv("GD2FS_IOTHREADS", 1))
        memthreads = int(os.getenv("GD2FS_MEMTHREADS", 1))
        streams = int(os.getenv("GD2FS_STREAMS", 1))

        _PROTECTED_KEYS = {
            "cpaddr",
            "dpaddr",
            "nics",
            "iothreads",
            "memthreads",
            "streams",
        }
        query = {
            k: v[0] if v else ""
            for k, v in urllib.parse.parse_qs(parsed.query).items()
            if k not in _PROTECTED_KEYS
        }

        return {
            "cpaddr": cpaddr,
            "dpaddr": dpaddr,
            "nics": nics,
            "iothreads": iothreads,
            "memthreads": memthreads,
            "streams": streams,
            **query,
        }, parsed.path

    def _build_mem_regions_iomem(self):
        for addr, bytes in self.mem_regions_per_layer:
            if self._get_iomem_by_address(addr, bytes) is not None:
                raise ValueError(
                    f"memory [{addr}, {addr + bytes}] is already registered"
                )

            iomem = self.client.RegIOMEM(addr, bytes)
            if iomem is None:
                raise RuntimeError(
                    f"cannot register memory [{addr}, {addr + bytes}] to GD2FS client"
                )

            bisect.insort(self.mem_regions_iomem, (addr, addr + bytes, iomem))

    def _get_iomem_by_address(self, addr: int, length: int) -> pygd2fs.IOMEM | None:
        idx = bisect.bisect_right(self.mem_regions_iomem, (addr, float("inf"))) - 1
        if idx < 0:
            return None

        start_addr, end_addr, iomem = self.mem_regions_iomem[idx]
        if not (start_addr <= addr <= end_addr - length + 1):
            return None

        return iomem

    def register_kvcache(self, config: ExOffloadingStorageKVCacheConfig) -> None:
        self.mem_regions_per_layer, self.block_bytes_per_layer = build_mem_regions(
            config
        )
        self._build_mem_regions_iomem()
        self.kvcache_config = config

    def _get_mem_regions(self, block_ids: list[int]) -> list[tuple[int, int]]:
        return get_mem_regions(
            self.mem_regions_per_layer, self.block_bytes_per_layer, block_ids
        )

    def _get_mem_tensors(self, block_ids: list[int]) -> list[torch.Tensor]:
        return get_mem_tensors(
            self.kvcache_config.kv_caches, block_ids, self.kvcache_config.split_k_and_v
        )

    def _create_sge_direct(self, mems: list[tuple[int, int]]) -> list[pygd2fs.SGE]:
        sgs = []
        for addr, size in mems:
            iomem = self._get_iomem_by_address(addr, size)
            assert iomem is not None, (
                f"memory region [{addr}, {addr + size}] not registered in GD2FS storage"
            )
            sgs.append(pygd2fs.SGE(addr, size, iomem))
        return sgs

    def _create_sge_bounce(
        self, mems: list[torch.Tensor]
    ) -> tuple[list[pygd2fs.SGE], torch.Tensor, Any]:
        host_data = torch.empty(
            tensors_total_numel(mems),
            dtype=mems[0].dtype,
            device="cpu",
            pin_memory=True,
        )

        length = host_data.element_size() * host_data.numel()
        iomem = self.client.RegIOMEM(host_data.data_ptr(), length)
        sgs = [pygd2fs.SGE(host_data.data_ptr(), length, iomem)]

        return sgs, host_data, iomem

    def _check_req_result(self, req: pygd2fs.Request | None, expect_length: int):
        if req is None:
            raise RuntimeError("GD2FS cannot initiate request")
        elif req.Status() != 0:
            raise RuntimeError(f"GD2FS request status is abnormal, {req}")
        elif req.Value() != expect_length:
            raise RuntimeError(
                f"GD2FS request length is abnormal, {req}, expect {expect_length}"
            )

    async def _load_direct(
        self, filepath: str, offset: int, block_ids: list[int]
    ) -> None:
        mem_regions = self._get_mem_regions(block_ids)
        sges = self._create_sge_direct(mem_regions)

        expect_length = sum(sge.length for sge in sges)

        req = await self.client.ReadAsync(filepath, offset, sges, 0)
        self._check_req_result(req, expect_length)

    async def _save_direct(
        self, filepath: str, offset: int, block_ids: list[int]
    ) -> None:
        mem_regions = self._get_mem_regions(block_ids)
        sges = self._create_sge_direct(mem_regions)

        expect_length = sum(sge.length for sge in sges)

        req = await self.client.WriteAsync(filepath, offset, sges, 0)
        self._check_req_result(req, expect_length)

    async def _load_bounce(
        self, filepath: str, offset: int, block_ids: list[int]
    ) -> None:
        mem_tensors = self._get_mem_tensors(block_ids)
        sges, host_tensor, iomem = self._create_sge_bounce(mem_tensors)

        expect_length = sum(sge.length for sge in sges)

        try:
            req = await self.client.ReadAsync(filepath, offset, sges, 0)
            self._check_req_result(req, expect_length)
            copy_data_h2d(host_tensor, mem_tensors)
        finally:
            self.client.DeregIOMEM(iomem)

    async def _save_bounce(
        self, filepath: str, offset: int, block_ids: list[int]
    ) -> None:
        mem_tensors = self._get_mem_tensors(block_ids)
        sges, host_tensor, iomem = self._create_sge_bounce(mem_tensors)
        copy_data_d2h(mem_tensors, host_tensor)

        expect_length = sum(sge.length for sge in sges)

        try:
            req = await self.client.WriteAsync(filepath, offset, sges, 0)
            self._check_req_result(req, expect_length)
        finally:
            self.client.DeregIOMEM(iomem)

    async def load(self, filepath: str, offset: int, block_ids: list[int]) -> None:
        if not block_ids:
            return

        await self._load_fn(filepath, offset, block_ids)

    async def save(self, filepath: str, offset: int, block_ids: list[int]) -> None:
        if not block_ids:
            return

        await self._save_fn(filepath, offset, block_ids)
