"""Small helpers for turning a bit array into a usable secret."""
import base64
import numpy as np


def convert_to_octets(bits: np.ndarray) -> bytes:
    """Pack a 0/1 array into bytes (pads with trailing zero bits if the
    length isn't a multiple of 8)."""
    bits = np.asarray(bits, dtype=int)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    byte_vals = np.packbits(bits)
    return byte_vals.tobytes()


def to_base64(bits: np.ndarray) -> str:
    return base64.b64encode(convert_to_octets(bits)).decode("ascii")
