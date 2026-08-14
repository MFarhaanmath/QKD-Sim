"""
BB84 core protocol logic, implemented with Cirq instead of Braket/Qiskit.

Mirrors the structure of the AWS sample's utils/bb84.py:
- initialize_protocol(): random bits + random bases for Alice/Bob
- encode_qubits(): Alice's state prep circuit (X for bit value, H for basis)
- measure_qubits(): Bob's basis-alignment + measurement circuit
- filter_qubits(): keeps only bits where Alice's and Bob's bases matched
"""
import numpy as np
import cirq


def initialize_protocol(number_of_qubits: int):
    """Generate random bit values and random encoding/measurement bases."""
    encoding_basis = np.random.randint(2, size=number_of_qubits)
    alice_states = np.random.randint(2, size=number_of_qubits)
    measurement_basis = np.random.randint(2, size=number_of_qubits)
    return encoding_basis, alice_states, measurement_basis


def encode_qubits(
    qubits: list, alice_states: np.ndarray, encoding_basis: np.ndarray
) -> cirq.Circuit:
    """Alice prepares her qubits: X gate if bit=1, then H if basis=1
    (diagonal basis)."""
    circuit = cirq.Circuit()
    for idx, q in enumerate(qubits):
        if alice_states[idx] == 1:
            circuit.append(cirq.X(q))
        if encoding_basis[idx] == 1:
            circuit.append(cirq.H(q))
    return circuit


def measure_qubits(
    circuit: cirq.Circuit, qubits: list, measurement_basis: np.ndarray
) -> cirq.Circuit:
    """Bob applies H on any qubit where he chose the diagonal basis, then
    measures every qubit."""
    for idx, q in enumerate(qubits):
        if measurement_basis[idx] == 1:
            circuit.append(cirq.H(q))
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit


def filter_qubits(
    bits: np.ndarray, encoding_basis: np.ndarray, measurement_basis: np.ndarray
) -> np.ndarray:
    """Keep only the bits where Alice's and Bob's bases matched (sifting)."""
    bits = np.asarray(bits)
    mask = encoding_basis == measurement_basis
    return bits[mask]


def run_round(
    number_of_qubits: int, bit_flip_probability: float, verbose: bool = False,
    round_number: int = None,
):
    """
    Run one full BB84 round: state prep -> noisy channel -> measurement
    -> sifting. Returns (alice_sifted, bob_sifted) as 0/1 numpy arrays.

    If verbose=True, prints Alice's bits/bases, Bob's bases/measured bits,
    which positions matched, and what survived sifting -- for this round.
    """
    encoding_basis_a, states_a, _ = initialize_protocol(number_of_qubits)
    _, _, measurement_basis_b = initialize_protocol(number_of_qubits)

    qubits = cirq.LineQubit.range(number_of_qubits)
    circuit = encode_qubits(qubits, states_a, encoding_basis_a)

    # Channel noise: bit-flip on every qubit, applied right after Alice's
    # state prep (mirrors the reference implementation's noise placement).
    noise = cirq.bit_flip(p=bit_flip_probability)
    circuit.append([noise.on(q) for q in qubits])

    circuit = measure_qubits(circuit, qubits, measurement_basis_b)

    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(circuit, repetitions=1)
    measured_bits = result.measurements["result"][0]

    match_mask = encoding_basis_a == measurement_basis_b
    alice_sifted = filter_qubits(states_a, encoding_basis_a, measurement_basis_b)
    bob_sifted = filter_qubits(measured_bits, encoding_basis_a, measurement_basis_b)

    if verbose:
        label = f"Round {round_number}" if round_number is not None else "Round"
        basis_name = {0: "Z", 1: "X"}
        a_basis_str = "".join(basis_name[b] for b in encoding_basis_a)
        b_basis_str = "".join(basis_name[b] for b in measurement_basis_b)
        match_str = "".join("v" if m else "." for m in match_mask)
        print(f"--- {label} ({number_of_qubits} qubits) ---")
        print(f"  Alice bits:    {array_to_string(states_a)}")
        print(f"  Alice bases:   {a_basis_str}   (Z=computational, X=diagonal)")
        print(f"  Bob bases:     {b_basis_str}")
        print(f"  Bob measured:  {array_to_string(measured_bits)}")
        print(f"  Basis match:   {match_str}")
        print(f"  Kept (sifted): Alice={array_to_string(alice_sifted)}  "
              f"Bob={array_to_string(bob_sifted)}  "
              f"({len(alice_sifted)}/{number_of_qubits} bits survived)\n")

    return alice_sifted, bob_sifted


def array_to_string(bits: np.ndarray) -> str:
    """Convert a 0/1 numpy array into a compact bitstring, for printing."""
    return "".join(str(int(b)) for b in bits)