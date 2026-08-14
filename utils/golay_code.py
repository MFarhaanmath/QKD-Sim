"""
[24,12,8] extended binary Golay code.

Construction: G = [I_12 | B], where B is the complement (including the
diagonal) of the adjacency matrix of the icosahedron graph. This is the
classic construction described in the standard references on the Golay
code (e.g. MacWilliams & Sloane). B is symmetric and satisfies B @ B = I
(mod 2), which is what makes the simplified single-sided syndrome
decoding trick below work.

This module has no dependency on any quantum SDK -- it's the same
regardless of whether you generate qubits with Cirq, Qiskit, or Braket.
"""
import numpy as np


def _build_b_matrix() -> np.ndarray:
    """Build the 12x12 B matrix from icosahedron vertex geometry."""
    phi = (1 + 5 ** 0.5) / 2
    verts = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            verts.append((0, s1 * 1, s2 * phi))
            verts.append((s1 * 1, s2 * phi, 0))
            verts.append((s2 * phi, 0, s1 * 1))
    verts = np.array(sorted(set(verts)))

    dist = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    adjacency = (np.abs(dist - min_dist) < 1e-6).astype(int)

    # Complement including the diagonal (adjacency diagonal is 0, so the
    # complement's diagonal becomes 1).
    b_matrix = (1 - adjacency).astype(int)
    return b_matrix


class GolayCode:
    """[24,12,8] extended binary Golay code: encode + syndrome decode."""

    def __init__(self):
        self._b_matrix = _build_b_matrix()
        self._generator_matrix = np.hstack(
            [np.eye(12, dtype=int), self._b_matrix]
        ) % 2
        # H = [B | I] so that codeword @ H.T == 0 (mod 2) for valid codewords.
        # parity_check is stored as H.T (24x12) so callers can do
        # `codeword @ parity_check % 2` directly, matching the reference
        # implementation's calling convention.
        h_matrix = np.hstack([self._b_matrix, np.eye(12, dtype=int)]) % 2
        self._parity_check = h_matrix.T % 2

    def get_generator_matrix(self) -> np.ndarray:
        """12x24 generator matrix G = [I | B]."""
        return self._generator_matrix

    def get_parity_check_matrix(self) -> np.ndarray:
        """24x12 matrix P such that `codeword @ P % 2` gives the syndrome."""
        return self._parity_check

    def get_b_matrix(self) -> np.ndarray:
        """12x12 B matrix, used to transform the syndrome into an error
        vector during decoding (relies on B @ B == I mod 2)."""
        return self._b_matrix

    def encode(self, message_bits: np.ndarray) -> np.ndarray:
        """Encode a 12-bit message into a 24-bit Golay codeword."""
        return (message_bits @ self._generator_matrix) % 2

    def decode_single_sided(
        self, received_message: np.ndarray, parity_bits: np.ndarray
    ):
        """
        Correct up to 3 errors, assuming errors occur ONLY in the 12
        message bits (the parity bits are assumed to have arrived over
        an error-free classical channel -- the standard BB84
        reconciliation assumption).

        Returns (corrected_message, success: bool).
        """
        codeword = np.concatenate([received_message, parity_bits]) % 2
        syndrome = (codeword @ self._parity_check) % 2
        error_estimate = (syndrome @ self._b_matrix) % 2

        if error_estimate.sum() > 3:
            return received_message, False

        corrected = (received_message + error_estimate) % 2
        return corrected, True
