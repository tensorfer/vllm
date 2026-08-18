# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# NFSv3 storage via libnfs (userspace client, no kernel mount). The kv_uri
# scheme "nfs" selects this storage; the server does an async page-cache warm
# when a client ACCESS(READ)es the file, so the scheduler's readahead step can
# touch the file on the NFS server before the load arrives.
import ctypes
import ctypes.util
import glob
import os
import sys
import threading
import urllib.parse

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    ExOffloadingStorage,
    ExOffloadingStorageKVCacheConfig,
    copy_data_d2h,
    copy_data_h2d,
    get_mem_tensors,
    tensors_total_numel,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_PORT = 12049

# Disable libnfs sync API warnings on Linux
os.environ.setdefault("NFS_DISABLE_WARN", "1")


def _load_libnfs():
    """Load libnfs shared library, supporting both macOS and Linux."""
    if sys.platform.startswith("linux"):
        # ldconfig knows where libnfs lives on any distro (Fedora: /usr/lib64,
        # Debian/Ubuntu: /usr/lib/<multiarch-triplet>/)
        found = ctypes.util.find_library("nfs")
        if found:
            return ctypes.CDLL(found, mode=ctypes.RTLD_GLOBAL)
        patterns = [
            "/usr/lib/*/libnfs.so*",
            "/usr/lib/libnfs.so*",
            "/usr/lib64/libnfs.so*",
            "/lib64/libnfs.so*",
        ]
    elif sys.platform == "darwin":
        patterns = [
            "/opt/homebrew/lib/libnfs*.dylib",
            "/usr/local/lib/libnfs*.dylib",
        ]
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    for pattern in patterns:
        hits = glob.glob(pattern)
        if hits:
            # Sort to pick the highest version if multiple exist
            hits.sort()
            return ctypes.CDLL(hits[-1], mode=ctypes.RTLD_GLOBAL)

    raise ImportError(
        "libnfs not found. "
        "On Linux: `apt install libnfs-dev` or `yum install libnfs`; "
        "on macOS: `brew install libnfs`"
    )


class _Libnfs:
    """Minimal ctypes binding for the libnfs sync API (NFSv3)."""

    def __init__(self):
        lib = _load_libnfs()
        self._lib = lib

        # nfs_init_context
        lib.nfs_init_context.restype = ctypes.c_void_p

        # nfs_parse_url_dir (may not exist on some Linux builds; we don't use it)
        if hasattr(lib, "nfs_parse_url_dir"):
            lib.nfs_parse_url_dir.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            lib.nfs_parse_url_dir.restype = ctypes.c_void_p

        # nfs_get_error
        lib.nfs_get_error.argtypes = [ctypes.c_void_p]
        lib.nfs_get_error.restype = ctypes.c_char_p

        # nfs_mount
        lib.nfs_mount.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.nfs_mount.restype = ctypes.c_int

        # nfs_open
        lib.nfs_open.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nfs_open.restype = ctypes.c_int

        # nfs_creat
        lib.nfs_creat.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.nfs_creat.restype = ctypes.c_int

        # nfs_pread
        lib.nfs_pread.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.nfs_pread.restype = ctypes.c_int

        # nfs_pwrite
        lib.nfs_pwrite.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.nfs_pwrite.restype = ctypes.c_int

        # nfs_close
        lib.nfs_close.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.nfs_close.restype = ctypes.c_int

    def mount(self, host, port, export):
        ctx = self._lib.nfs_init_context()
        if not ctx:
            raise RuntimeError("nfs_init_context failed")

        # nfs_parse_url_dir tells libnfs the NFS and mount ports (both services
        # run on the same port in this deployment). Distro builds export it;
        # some minimal Linux builds don't.
        if hasattr(self._lib, "nfs_parse_url_dir"):
            url = f"nfs://{host}:{port}/?mountport={port}&auto-traverse-mounts=0".encode()
            if not self._lib.nfs_parse_url_dir(ctx, url):
                raise RuntimeError(f"parse url failed: {self._lib.nfs_get_error(ctx)}")

        if self._lib.nfs_mount(ctx, host.encode(), export.encode()) != 0:
            raise RuntimeError(f"nfs mount failed: {self._lib.nfs_get_error(ctx)}")
        return ctx

    def open(self, ctx, path, flags):
        fh = ctypes.c_void_p()
        if self._lib.nfs_open(ctx, path.encode(), flags, ctypes.byref(fh)) != 0:
            err = self._lib.nfs_get_error(ctx)
            # The file may not exist yet (the server creates session files via
            # accellm's Create); create it, then retry the open.
            if b"NOENT" in err or b"ENOENT" in err:
                cfh = ctypes.c_void_p()
                if self._lib.nfs_creat(ctx, path.encode(), 0o644,
                                       ctypes.byref(cfh)) != 0:
                    raise OSError(f"nfs_creat {path}: {self._lib.nfs_get_error(ctx)}")
                self._lib.nfs_close(ctx, cfh)
                if self._lib.nfs_open(ctx, path.encode(), flags,
                                      ctypes.byref(fh)) != 0:
                    raise OSError(f"nfs_open {path}: {self._lib.nfs_get_error(ctx)}")
            else:
                raise OSError(f"nfs_open {path}: {err}")
        return fh

    def pread(self, ctx, fh, size, offset):
        # nfs_pread caps each call at the transport's max transfer size; loop
        # until the whole range is read.
        buf = (ctypes.c_ubyte * size)()
        total = 0
        while total < size:
            n = self._lib.nfs_pread(
                ctx, fh, ctypes.byref(buf, total),
                size - total, offset + total
            )
            if n <= 0:
                raise OSError(f"short read at {offset + total}: got {n} bytes")
            total += n
        return bytes(buf)

    def pwrite(self, ctx, fh, data, offset):
        # nfs_pwrite caps each call at the transport's max transfer size; loop
        # until all bytes are written.
        buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        total = 0
        while total < len(data):
            n = self._lib.nfs_pwrite(
                ctx, fh, ctypes.byref(buf, total),
                len(data) - total, offset + total
            )
            if n <= 0:
                raise OSError(f"short write at {offset + total}: got {n} bytes")
            total += n

    def close(self, ctx, fh):
        self._lib.nfs_close(ctx, fh)


class _NFSConnection:
    """One mounted libnfs context per (host, port), shared across storages."""

    _libnfs = None
    _lock = threading.RLock()  # reentrant: get() nests _lib()
    _contexts = {}

    @classmethod
    def _lib(cls):
        with cls._lock:
            if cls._libnfs is None:
                cls._libnfs = _Libnfs()
            return cls._libnfs

    @classmethod
    def get(cls, host, port):
        key = (host, port)
        with cls._lock:
            if key not in cls._contexts:
                lib = cls._lib()
                ctx = lib.mount(host, port, "/")
                logger.info("nfs storage: mounted %s:%d", host, port)
                cls._contexts[key] = (lib, ctx)
            return cls._contexts[key]


class NFSStorage(ExOffloadingStorage):
    def __init__(self, extra_config: dict[str, str]):
        self.host = extra_config.get("host")
        self.port = int(extra_config.get("port", _DEFAULT_PORT))
        self.root_path = extra_config.get("root_path")

    @classmethod
    def parse_uri(cls, uri: str) -> tuple[dict[str, str], str]:
        if not uri.startswith("nfs://"):
            raise ValueError("invalid NFS URI format")

        parsed = urllib.parse.urlparse(uri)
        filepath = parsed.path.lstrip("/")
        return {
            "host": parsed.hostname,
            "port": str(parsed.port or _DEFAULT_PORT),
        }, filepath

    def register_kvcache(self, config: ExOffloadingStorageKVCacheConfig) -> None:
        self.kvcache_config = config

    def _pread(self, path, offset, size):
        lib, ctx = _NFSConnection.get(self.host, self.port)
        fh = lib.open(ctx, path, os.O_RDONLY)
        try:
            return lib.pread(ctx, fh, size, offset)
        finally:
            lib.close(ctx, fh)

    def _pwrite(self, path, offset, data):
        lib, ctx = _NFSConnection.get(self.host, self.port)
        fh = lib.open(ctx, path, os.O_RDWR)
        try:
            lib.pwrite(ctx, fh, data, offset)
        finally:
            lib.close(ctx, fh)

    async def load(self, filepath: str, offset: int, block_ids: list[int]) -> None:
        if not block_ids:
            return

        mem_tensors = get_mem_tensors(
            self.kvcache_config.kv_caches, block_ids, self.kvcache_config.split_k_and_v
        )
        host_buffer = torch.empty(
            tensors_total_numel(mem_tensors), dtype=mem_tensors[0].dtype, device="cpu"
        )

        buf = memoryview(host_buffer.flatten().view(torch.uint8).numpy())
        data = self._pread(filepath, offset, len(buf))
        buf[:] = data

        copy_data_h2d(host_buffer, mem_tensors)

    async def save(self, filepath: str, offset: int, block_ids: list[int]) -> None:
        if not block_ids:
            return

        mem_tensors = get_mem_tensors(
            self.kvcache_config.kv_caches, block_ids, self.kvcache_config.split_k_and_v
        )
        host_buffer = torch.empty(
            tensors_total_numel(mem_tensors), dtype=mem_tensors[0].dtype, device="cpu"
        )
        copy_data_d2h(mem_tensors, host_buffer)

        buf = memoryview(host_buffer.flatten().view(torch.uint8).numpy())
        self._pwrite(filepath, offset, buf)
