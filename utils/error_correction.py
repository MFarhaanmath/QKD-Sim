"""
Key reconciliation: Alice encodes her raw key with the Golay code and
sends only the parity bits to Bob (over an assumed-authenticated
classical channel). Bob uses those parity bits plus his own raw key to
correct up to 3 bit errors without either party ever transmitting the
actual key bits.
"""
import numpy as np
from utils.golay_code import GolayCode


def alice_side(golay: GolayCode, alice_raw_key: np.ndarray):
    """Returns (parity_bits_to_send, syndrome_should_be_zero)."""
    codeword = golay.encode(alice_raw_key)
    syndrome = (codeword @ golay.get_parity_check_matrix()) % 2
    parity_bits = codeword[12:]
    return parity_bits, syndrome


def bob_side(golay: GolayCode, bob_raw_key: np.ndarray, parity_bits: np.ndarray):
    """Returns (corrected_key, success: bool)."""
    return golay.decode_single_sided(bob_raw_key, parity_bits)
