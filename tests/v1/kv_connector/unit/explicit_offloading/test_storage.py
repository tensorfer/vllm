# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.explicit_offloading.storage.abstract import (  # noqa: E501
    copy_data_d2h,
    copy_data_h2d,
    copy_data_split_chunks,
)

pytestmark = pytest.mark.cpu_test

DTYPES = [torch.float32, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
def test_copy_data_h2d_defaults_copy_all_host_data(dtype):
    """Default args keep the original semantics: the whole host tensor is
    copied sequentially into the flattened destination tensors."""
    host = torch.arange(8, dtype=dtype)
    dev = [
        torch.full((2, 2), -1, dtype=dtype),
        torch.full((2, 2), -1, dtype=dtype),
    ]

    assert copy_data_h2d(host, dev) == (2, 0)
    assert dev[0].flatten().tolist() == [0, 1, 2, 3]
    assert dev[1].flatten().tolist() == [4, 5, 6, 7]


@pytest.mark.parametrize("dtype", DTYPES)
def test_copy_data_h2d_host_bytes_limits_copy(dtype):
    """Only host_bytes of host data is copied; the rest of the destination
    tensors is left untouched."""
    host = torch.arange(8, dtype=dtype)
    dev = [
        torch.full((4,), -1, dtype=dtype),
        torch.full((4,), -1, dtype=dtype),
    ]

    assert copy_data_h2d(host, dev, host_bytes=host.element_size() * 2) == (0, 2)
    assert dev[0].tolist() == [0, 1, -1, -1]
    assert dev[1].tolist() == [-1, -1, -1, -1]


def test_copy_data_h2d_ignores_host_data_beyond_dev_capacity():
    host = torch.arange(10, dtype=torch.float32)
    dev = [torch.full((4,), -1.0), torch.full((4,), -1.0)]

    assert copy_data_h2d(host, dev) == (2, 0)
    assert dev[0].tolist() == [0, 1, 2, 3]
    assert dev[1].tolist() == [4, 5, 6, 7]


def test_copy_data_h2d_copies_from_dev_index_and_offset():
    host = torch.arange(2, dtype=torch.float32)
    dev = [torch.full((4,), -1.0), torch.full((4,), -1.0)]

    assert copy_data_h2d(host, dev, dev_index=1, dev_off=2) == (2, 0)
    assert dev[0].tolist() == [-1.0, -1.0, -1.0, -1.0]
    assert dev[1].tolist() == [-1.0, -1.0, 0.0, 1.0]


def test_copy_data_h2d_chunked_rounds_fill_like_single_copy():
    """Feeding the returned dev_index/dev_off into the next round makes
    chunked copies land exactly like a single full copy."""
    expected = torch.arange(12, dtype=torch.float32)
    one_shot = [torch.full((6,), -1.0), torch.full((6,), -1.0)]
    copy_data_h2d(expected, one_shot)

    chunked = [torch.full((6,), -1.0), torch.full((6,), -1.0)]
    index, offset = 0, 0
    for start in range(0, 12, 4):
        chunk = expected[start : start + 4]
        index, offset = copy_data_h2d(
            chunk, chunked, index, offset, host_bytes=chunk.element_size() * 4
        )

    assert (index, offset) == (2, 0)
    assert torch.equal(chunked[0], one_shot[0])
    assert torch.equal(chunked[1], one_shot[1])


def test_copy_data_h2d_copy_false_advances_position_only():
    host = torch.arange(2, dtype=torch.float32)
    dev = [torch.full((4,), -1.0), torch.full((4,), -1.0)]

    assert copy_data_h2d(host, dev, dev_off=2, host_bytes=8, copy=False) == (1, 0)
    assert dev[0].tolist() == [-1.0, -1.0, -1.0, -1.0]
    assert dev[1].tolist() == [-1.0, -1.0, -1.0, -1.0]


def test_copy_data_h2d_empty_dev_list_is_noop():
    host = torch.arange(4, dtype=torch.float32)
    assert copy_data_h2d(host, [], dev_index=1, dev_off=3) == (1, 3)


@pytest.mark.parametrize("dtype", DTYPES)
def test_copy_data_d2h_defaults_copy_all_dev_data(dtype):
    """Default args keep the original semantics: all source tensors are
    copied sequentially into the flat host tensor."""
    data = [
        torch.arange(4, dtype=dtype).reshape(2, 2),
        torch.arange(4, 8, dtype=dtype).reshape(2, 2),
    ]
    host = torch.full((8,), -1, dtype=dtype)

    assert copy_data_d2h(host, data) == (2, 0)
    assert host.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("dtype", DTYPES)
def test_copy_data_d2h_host_bytes_limits_copy(dtype):
    """Only host_bytes are written into host_data; the rest of host_data
    is left untouched."""
    data = [
        torch.arange(4, dtype=dtype),
        torch.arange(4, 8, dtype=dtype),
    ]
    host = torch.full((8,), -1, dtype=dtype)

    assert copy_data_d2h(host, data, host_bytes=host.element_size() * 2) == (0, 2)
    assert host.tolist() == [0, 1, -1, -1, -1, -1, -1, -1]


def test_copy_data_d2h_ignores_dev_data_beyond_host_capacity():
    data = [
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, 10, dtype=torch.float32),
    ]
    host = torch.full((8,), -1.0)

    assert copy_data_d2h(host, data) == (1, 0)
    assert host.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_copy_data_d2h_copies_from_dev_index_and_offset():
    data = [
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, 8, dtype=torch.float32),
    ]
    host = torch.full((4,), -1.0)

    assert copy_data_d2h(host, data, dev_index=1, dev_off=2) == (2, 0)
    assert host.tolist() == [6.0, 7.0, -1.0, -1.0]


def test_copy_data_d2h_chunked_rounds_fill_like_single_copy():
    """Feeding the returned dev_index/dev_off into the next round makes
    chunked copies land exactly like a single full copy."""
    expected = torch.arange(12, dtype=torch.float32)
    data = [expected[:6], expected[6:]]

    one_shot = torch.full((12,), -1.0)
    copy_data_d2h(one_shot, data)

    chunked = torch.full((12,), -1.0)
    index, offset = 0, 0
    for start in range(0, 12, 4):
        host_chunk = torch.full((4,), -1.0)
        index, offset = copy_data_d2h(
            host_chunk,
            data,
            index,
            offset,
            host_bytes=host_chunk.element_size() * 4,
        )
        chunked[start : start + 4] = host_chunk

    assert (index, offset) == (2, 0)
    assert torch.equal(chunked, one_shot)


def test_copy_data_d2h_copy_false_advances_position_only():
    data = [
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, 8, dtype=torch.float32),
    ]
    host = torch.full((4,), -1.0)

    assert copy_data_d2h(host, data, dev_off=2, host_bytes=8, copy=False) == (1, 0)
    assert host.tolist() == [-1.0, -1.0, -1.0, -1.0]


def test_copy_data_d2h_empty_dev_list_is_noop():
    host = torch.arange(4, dtype=torch.float32)
    assert copy_data_d2h(host, [], dev_index=1, dev_off=3) == (1, 3)


def test_copy_data_split_chunks_aligned_offset():
    """An aligned offset splits into host-sized chunks, all aligned."""
    host = torch.empty(4, dtype=torch.float32)
    dev = [torch.full((6,), -1.0), torch.full((6,), -1.0)]

    assert copy_data_split_chunks(host, dev, 0, 8, copy_data_h2d) == [
        (0, 16, 0, 0),
        (16, 16, 0, 4),
        (32, 16, 1, 2),
    ]


def test_copy_data_split_chunks_unaligned_offset_trims_first_chunk():
    """The first chunk is trimmed to the next alignment boundary so every
    following chunk starts aligned."""
    host = torch.empty(4, dtype=torch.float32)
    dev = [torch.full((6,), -1.0), torch.full((6,), -1.0)]

    chunks = copy_data_split_chunks(host, dev, 4, 8, copy_data_h2d)

    assert chunks == [
        (4, 4, 0, 0),
        (8, 16, 0, 1),
        (24, 16, 0, 5),
        (40, 12, 1, 3),
    ]
    assert all(c[0] % 8 == 0 for c in chunks[1:])


def test_copy_data_split_chunks_same_for_h2d_and_d2h():
    host = torch.empty(4, dtype=torch.float32)
    dev = [torch.full((6,), -1.0), torch.full((6,), -1.0)]

    assert copy_data_split_chunks(
        host, dev, 4, 8, copy_data_h2d
    ) == copy_data_split_chunks(host, dev, 4, 8, copy_data_d2h)


def test_copy_data_split_chunks_total_smaller_than_host():
    host = torch.empty(16, dtype=torch.float32)
    dev = [torch.full((3,), -1.0)]

    assert copy_data_split_chunks(host, dev, 0, 8, copy_data_h2d) == [(0, 12, 0, 0)]


def test_copy_data_split_chunks_empty_dev_list():
    host = torch.empty(4, dtype=torch.float32)
    assert copy_data_split_chunks(host, [], 0, 8, copy_data_h2d) == []


@pytest.mark.parametrize(
    "offset,align,chunk_len,sizes",
    [
        (0, 8, 16, [6, 6]),
        (4, 8, 16, [6, 6]),
        (16, 8, 16, [4, 4, 4]),
        (0, 16, 32, [10, 2]),
        (8, 16, 32, [7]),
        (12, 4, 8, [1, 2, 3, 5]),
    ],
)
def test_copy_data_split_chunks_covers_file_range_aligned(
    offset, align, chunk_len, sizes
):
    """Chunks tile the file range contiguously, never exceed the host tensor
    size and keep every offset but the first one aligned."""
    host = torch.empty(chunk_len // 4, dtype=torch.float32)
    dev = [torch.full((s,), -1.0) for s in sizes]

    chunks = copy_data_split_chunks(host, dev, offset, align, copy_data_h2d)

    total = 4 * sum(sizes)
    assert sum(c[1] for c in chunks) == total
    file_off = offset
    for c_off, c_len, _, _ in chunks:
        assert c_off == file_off
        assert 0 < c_len <= chunk_len
        file_off += c_len
    assert all(c[0] % align == 0 for c in chunks[1:])


def test_copy_data_split_chunks_positions_feed_real_h2d_copies():
    """Chunk positions fed into real copies land the file data exactly like
    a single full copy (the gd2fs load bounce path)."""
    file_data = torch.arange(25, dtype=torch.float32)
    offset = 4
    slot = torch.empty(4, dtype=torch.float32)
    dev = [torch.full((6,), -1.0), torch.full((14,), -1.0)]

    chunks = copy_data_split_chunks(slot, dev, offset, 8, copy_data_h2d)
    for c_off, c_len, dev_index, dev_off in chunks:
        host = file_data[c_off // 4 : (c_off + c_len) // 4]
        copy_data_h2d(host, dev, dev_index, dev_off, host_bytes=c_len)

    flat = torch.cat([d.flatten() for d in dev])
    assert torch.equal(flat, file_data[1:21])


def test_copy_data_split_chunks_positions_feed_real_d2h_copies():
    """Chunk positions fed into real copies land the device data exactly like
    a single full copy (the gd2fs save bounce path)."""
    data = [
        torch.arange(6, dtype=torch.float32),
        torch.arange(6, 12, dtype=torch.float32),
    ]
    expected = torch.cat([d.flatten() for d in data])
    host_file = torch.full((25,), -1.0)
    offset = 4
    slot = torch.empty(4, dtype=torch.float32)

    chunks = copy_data_split_chunks(slot, data, offset, 8, copy_data_d2h)
    for c_off, c_len, dev_index, dev_off in chunks:
        host = host_file[c_off // 4 : (c_off + c_len) // 4]
        copy_data_d2h(host, data, dev_index, dev_off, host_bytes=c_len)

    assert torch.equal(host_file[1:13], expected)
    assert host_file[:1].tolist() == [-1.0]
    assert host_file[13:].tolist() == [-1.0] * 12
