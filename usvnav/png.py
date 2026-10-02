"""Minimal PNG writer, stdlib only.

The scoring machine's bundle should not grow an image library for the sake of writing
debug pictures, and the replay tool (`T1-RES-12`) has to produce something a human can
open. Truecolour 8-bit, no interlacing, one `IDAT` -- that is the whole format we need.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def write_png(path, rgb: np.ndarray, *, scale: int = 1) -> str:
    """Write an (H, W, 3) uint8 array. `scale` upsamples by nearest neighbour.

    Nearest neighbour, never interpolation: an upscaled raster is used to inspect the
    anti-aliased edges (4-V8), and a smoothing filter would invent the very sub-pixel
    detail we are looking at.
    """
    with open(path, "wb") as f:
        f.write(encode_png(rgb, scale=scale))
    return str(path)


def encode_png(rgb: np.ndarray, *, scale: int = 1) -> bytes:
    """The PNG file as bytes; `write_png` without the file. The studio serves these."""
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got {a.shape}")
    if scale > 1:
        a = np.repeat(np.repeat(a, scale, axis=0), scale, axis=1)
    h, w, _ = a.shape

    stride = w * 3
    raw = np.empty((h, stride + 1), dtype=np.uint8)
    raw[:, 0] = 0                                  # filter type 0 (None) per scanline
    raw[:, 1:] = a.reshape(h, stride)

    header = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
            + _chunk(b"IEND", b""))
