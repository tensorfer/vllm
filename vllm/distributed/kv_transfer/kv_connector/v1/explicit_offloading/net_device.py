# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import os
import random
import re
from enum import IntEnum, auto

import psutil
import torch

from vllm.distributed.parallel_state import get_world_group


def get_device_class(pci_path: str | None):
    if not pci_path:
        return "unknown"

    device_class_path = os.path.join(pci_path, "class")
    try:
        with open(device_class_path) as f:
            return f.read().strip()
    except Exception:
        return "unknown"


class DeviceInfo:
    class DistanceLevel(IntEnum):
        SELF = 0
        PIX = auto()
        PXB = auto()
        PHB = auto()
        NODE = auto()
        SYS = auto()
        UNKNOWN = auto()

    def __init__(self, name: str, path: str):
        self.name = name
        self._path = path

        self.real_path: str | None = None
        self.numa_node: int | None = None
        self.pci_host_bridge: str | None = None
        self.pci_switch: str | None = None

        self._get_real_path()
        self._get_numa_node()
        self._get_pci_host_bridge()
        self._get_pci_switch()

    def __repr__(self) -> str:
        return (
            f"Device {self.name}:\n"
            f"  real_path: {self.real_path}\n"
            f"  numa_node: {self.numa_node}\n"
            f"  pci_host_bridge: {self.pci_host_bridge}\n"
            f"  pci_switch: {self.pci_switch}"
        )

    def _get_real_path(self):
        if os.path.exists(self._path):
            self.real_path = os.path.realpath(self._path)
            assert self.real_path.startswith("/sys/devices/pci")

    def _get_numa_node(self):
        numa_node_path = os.path.join(self._path, "numa_node")
        if os.path.exists(numa_node_path):
            with open(numa_node_path) as f:
                val = f.read().strip()
                if val and val != "-1":
                    self.numa_node = int(val)

    def _get_pci_host_bridge(self):
        if not self.real_path:
            return

        m = re.match(r"(/sys/devices/pci[0-9a-fA-F:]+)", self.real_path)
        if m:
            self.pci_host_bridge = m.group(1)

    def _get_pci_switch(self):
        if not self.real_path:
            return

        self.pci_switch = os.path.dirname(self.real_path)
        device_class = get_device_class(self.pci_switch)
        # linux/include/linux/pci_ids.h
        # #define PCI_CLASS_BRIDGE_PCI            0x0604
        # #define PCI_CLASS_BRIDGE_PCI_NORMAL             0x060400
        # #define PCI_CLASS_BRIDGE_PCI_SUBTRACTIVE        0x060401
        if device_class not in ["0x0604", "0x060400", "0x060401"]:
            raise ValueError(f"device({self.pci_switch}) is not pci switch")

    def _common_path(self, other: DeviceInfo) -> str:
        if not self.real_path or not other.real_path:
            raise ValueError("device real path is empty")
        return os.path.commonpath([self.real_path, other.real_path])

    def _numa_node_equal_to(self, other: DeviceInfo) -> bool:
        return (
            self.numa_node is not None
            and other.numa_node is not None
            and self.numa_node == other.numa_node
        )

    def _pci_host_bridge_equal_to(self, other: DeviceInfo) -> bool:
        return (
            bool(self.pci_host_bridge)
            and bool(other.pci_host_bridge)
            and self.pci_host_bridge == other.pci_host_bridge
        )

    def _pci_switch_equal_to(self, other: DeviceInfo) -> bool:
        return (
            bool(self.pci_switch)
            and bool(other.pci_switch)
            and self.pci_switch == other.pci_switch
        )

    def _pci_path_distance_to(self, other: DeviceInfo) -> int:
        try:
            common = self._common_path(other)
            assert self.real_path and other.real_path
            rel1 = os.path.relpath(self.real_path, common)
            rel2 = os.path.relpath(other.real_path, common)
            steps1 = len(rel1.split(os.sep)) if rel1 != "." else 0
            steps2 = len(rel2.split(os.sep)) if rel2 != "." else 0
            return steps1 + steps2
        except Exception:
            return -1

    def distance(self, other: DeviceInfo) -> tuple[DistanceLevel, int]:
        if self.real_path is None or other.real_path is None:
            return self.DistanceLevel.UNKNOWN, -1

        if self.real_path == other.real_path:
            return self.DistanceLevel.SELF, 0

        distance = self._pci_path_distance_to(other)

        if not self._pci_host_bridge_equal_to(other):
            if self.numa_node is None or other.numa_node is None:
                return self.DistanceLevel.UNKNOWN, -1

            if not self._numa_node_equal_to(other):
                return self.DistanceLevel.SYS, distance

            return self.DistanceLevel.NODE, distance

        if self._common_path(other) == self.pci_host_bridge:
            return self.DistanceLevel.PHB, distance

        if self._pci_switch_equal_to(other):
            return self.DistanceLevel.PIX, distance

        return self.DistanceLevel.PXB, distance


def get_current_device_info() -> DeviceInfo:
    device = get_world_group().device
    device_module = torch.get_device_module(device)
    if device_module == torch.cuda:
        props = torch.cuda.get_device_properties(device)
        bus_id = (
            f"{props.pci_domain_id:04x}:"
            f"{props.pci_bus_id:02x}:"
            f"{props.pci_device_id:02x}.0"
        )
        device_name = f"{device}({props.uuid})"
        return DeviceInfo(name=device_name, path=f"/sys/bus/pci/devices/{bus_id}")
    else:
        raise RuntimeError(f"Not support {device_module}")


def get_nic_name_dict(addr_list: list[str]) -> dict[str, list[str]]:
    nic_dict = {}
    addrs = psutil.net_if_addrs()
    for iface_name, iface_addrs in addrs.items():
        matched = [addr.address for addr in iface_addrs if addr.address in addr_list]
        if matched:
            nic_dict[iface_name] = matched
    return nic_dict


def get_nic_info_dict(nic_name_list: list[str]) -> dict[str, DeviceInfo]:
    nic_info_dict = {}
    for nic_name in nic_name_list:
        nic_path = f"/sys/class/net/{nic_name}/device"
        nic_info_dict[nic_name] = DeviceInfo(nic_name, nic_path)
    return nic_info_dict


def find_closest_nics(
    dev: DeviceInfo, nics: dict[str, DeviceInfo], n: int = 1, n_random: bool = False
) -> list[tuple[str, DeviceInfo.DistanceLevel, int]]:
    candidates: list[tuple[str, DeviceInfo.DistanceLevel, int, float]] = []
    n = len(nics) if n <= 0 else n
    r = random.Random()

    for name, nic in nics.items():
        level, distance = dev.distance(nic)
        if level in [DeviceInfo.DistanceLevel.SELF, DeviceInfo.DistanceLevel.UNKNOWN]:
            continue
        candidates.append((name, level, distance, r.random()))

    candidates.sort(key=lambda x: (x[1], x[2], x[3] if n_random else x[0]))

    return [(name, level, distance) for name, level, distance, _ in candidates[:n]]
