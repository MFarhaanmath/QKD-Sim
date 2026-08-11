"""
BB84 with noise + Golay [24,12,8] key reconciliation, implemented in Cirq.

Run:
    python main.py
"""
import numpy as np

from utils.bb84_cirq import run_round, array_to_string
from utils.golay_code import GolayCode
from utils.error_correction import alice_side, bob_side
from utils.secret_utils import to_base64

BIT_FLIP_PROBABILITY = 0.1
NUMBER_OF_QUBITS = 12
ERROR_CORRECTION_CHUNK_SIZE = 12
VERBOSE = True  # set False to go back to the quiet summary-only output


def generate_sifted_key(chunk_size: int, bit_flip_probability: float, verbose: bool = False):
    """Repeat BB84 rounds until we have `chunk_size` sifted bits."""
    alice_key = np.array([], dtype=int)
    bob_key = np.array([], dtype=int)
    round_number = 0

    while len(alice_key) < chunk_size:
        round_number += 1
        alice_bits, bob_bits = run_round(
            NUMBER_OF_QUBITS, bit_flip_probability,
            verbose=verbose, round_number=round_number,
        )
        alice_key = np.concatenate([alice_key, alice_bits])
        bob_key = np.concatenate([bob_key, bob_bits])
        if verbose:
            print(f"  Running total: {len(alice_key)}/{chunk_size} sifted bits\n")

    return alice_key[:chunk_size], bob_key[:chunk_size]


def main():
    print(f"Generating a {ERROR_CORRECTION_CHUNK_SIZE}-bit sifted key block "
          f"(bit-flip probability = {BIT_FLIP_PROBABILITY})...\n")

    alice_raw_key, bob_raw_key = generate_sifted_key(
        ERROR_CORRECTION_CHUNK_SIZE, BIT_FLIP_PROBABILITY, verbose=VERBOSE
    )

    print(f"Alice's raw key: {array_to_string(alice_raw_key)}")
    print(f"Bob's raw key:   {array_to_string(bob_raw_key)}")

    mismatches = int((alice_raw_key != bob_raw_key).sum())
    print(f"Mismatched bits before reconciliation: {mismatches}\n")

    golay = GolayCode()
    parity_bits, alice_syndrome = alice_side(golay, alice_raw_key)
    print(f"Parity bits sent to Bob: {array_to_string(parity_bits)}")
    print(f"Alice's syndrome (should be all zero): "
          f"{array_to_string(alice_syndrome)}\n")

    corrected_key, success = bob_side(golay, bob_raw_key, parity_bits)

    if not success:
        print("Reconciliation FAILED: more than 3 errors in this block. "
              "Discard and retry with a fresh block.")
        return

    print(f"Bob's corrected key: {array_to_string(corrected_key)}")
    match = bool((corrected_key == alice_raw_key).all())
    print(f"Keys match after reconciliation: {match}")

    if match:
        print(f"\nFinal shared secret key (base64): {to_base64(alice_raw_key)}")


if __name__ == "__main__":
    main()