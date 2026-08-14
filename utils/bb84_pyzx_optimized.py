import math
import numpy as np
import cirq
from cirq_qiskit import from_cirq
from qiskit import transpile, QuantumCircuit
import pyzx as zx

# --- Your BB84 helpers (unchanged) ---
def initialize_protocol(number_of_qubits: int):
    encoding_basis = np.random.randint(2, size=number_of_qubits)
    alice_states = np.random.randint(2, size=number_of_qubits)
    measurement_basis = np.random.randint(2, size=number_of_qubits)
    return encoding_basis, alice_states, measurement_basis

def encode_qubits(qubits: list, alice_states: np.ndarray, encoding_basis: np.ndarray) -> cirq.Circuit:
    c = cirq.Circuit()
    for idx, q in enumerate(qubits):
        if alice_states[idx] == 1:
            c.append(cirq.X(q))
        if encoding_basis[idx] == 1:
            c.append(cirq.H(q))
    return c

def measure_qubits(circuit: cirq.Circuit, qubits: list, measurement_basis: np.ndarray) -> cirq.Circuit:
    for idx, q in enumerate(qubits):
        if measurement_basis[idx] == 1:
            circuit.append(cirq.H(q))
    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit

def filter_qubits(bits: np.ndarray, encoding_basis: np.ndarray, measurement_basis: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits)
    mask = encoding_basis == measurement_basis
    return bits[mask]

def array_to_string(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in bits)

# --- Utility: convert a pyzx.Circuit (extracted optimized circuit) to a cirq.Circuit ---
def pyzx_to_cirq(pyzx_circ: zx.Circuit, cirq_qubits: list) -> cirq.Circuit:
    """
    Best-effort conversion of common pyzx gates back to Cirq operations.
    Supports: CNOT, CZ, H/HAD, S, T, T* (T dagger), X, Y, Z, ZPhase(angle), XPhase(angle).
    If you see "Unsupported gate" errors, paste c_opt.to_qasm() here or tell me the gate names.
    """
    c = cirq.Circuit()
    for g in pyzx_circ.gates:
        name = getattr(g, "name", None)
        # some pyzx gate objects store fields differently; try common attributes
        control = getattr(g, "control", None)
        target = getattr(g, "target", None)
        # many gates use .targets or .controls lists; try those too
        targets = getattr(g, "targets", None) or getattr(g, "qubits", None)
        params = getattr(g, "params", None) or getattr(g, "phase", None)
        # Map common gates:
        if name in ("CNOT", "CX"):
            c.append(cirq.CNOT(cirq_qubits[control], cirq_qubits[target]))
        elif name == "CZ":
            c.append(cirq.CZ(cirq_qubits[control], cirq_qubits[target]))
        elif name in ("H", "HAD", "HADAMARD"):
            # might be a single-target gate: use .target if present else first element of targets
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.H(cirq_qubits[t]))
        elif name == "S":
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.S(cirq_qubits[t]))
        elif name == "T":
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.T(cirq_qubits[t]))
        elif name == "T*":  # T dagger (adjoint)
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.T(cirq_qubits[t])**-1)
        elif name == "X":
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.X(cirq_qubits[t]))
        elif name == "Y":
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.Y(cirq_qubits[t]))
        elif name == "Z":
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.Z(cirq_qubits[t]))
        elif name in ("ZPhase", "RZ"):
            # param angle expected in radians (pyzx often stores numeric radians)
            angle = None
            if isinstance(params, (list, tuple)) and len(params) > 0:
                angle = float(params[0])
            elif isinstance(params, (int, float)):
                angle = float(params)
            else:
                # try attribute .phase if present (might be Fraction*PI)
                angle = getattr(g, "phase", None)
                if angle is not None:
                    angle = float(angle)
            if angle is None:
                raise NotImplementedError(f"ZPhase gate without parameter in gate {g}")
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.rz(angle)(cirq_qubits[t]))
        elif name in ("XPhase", "RX"):
            angle = None
            if isinstance(params, (list, tuple)) and len(params) > 0:
                angle = float(params[0])
            elif isinstance(params, (int, float)):
                angle = float(params)
            else:
                angle = getattr(g, "phase", None)
                if angle is not None:
                    angle = float(angle)
            if angle is None:
                raise NotImplementedError(f"XPhase gate without parameter in gate {g}")
            t = target if target is not None else (targets[0] if targets else None)
            c.append(cirq.rx(angle)(cirq_qubits[t]))
        else:
            # Last attempt: many pyzx gate objects implement to_quipper/to_qasm; use name-based fallback:
            raise NotImplementedError(f"Unsupported/unknown pyzx gate: {name}. "
                                      "If you paste c_opt.to_qasm() I can add a mapping.")
    return c

# --- Main: isolate unitary, optimize, stitch back ---
def optimize_bb84_unitary_and_run(number_of_qubits: int, bit_flip_probability: float = 0.0, verbose: bool = True):
    # 1) initialize protocol bits/bases
    encoding_basis_a, states_a, _ = initialize_protocol(number_of_qubits)
    _, _, measurement_basis_b = initialize_protocol(number_of_qubits)

    # 2) build Cirq qubits and the various circuit parts
    qubits = list(cirq.LineQubit.range(number_of_qubits))

    # Alice's unitary state preparation (X if bit=1; H if encoding basis=1)
    prep_circ = encode_qubits(qubits, states_a, encoding_basis_a)

    # Bob's basis-alignment unitary (H on qubits he measures in diagonal basis)
    bob_unitary_circ = cirq.Circuit()
    for idx, q in enumerate(qubits):
        if measurement_basis_b[idx] == 1:
            bob_unitary_circ.append(cirq.H(q))

    # Noise layer (a non-unitary channel) — we keep it in Cirq only
    noise_ops = [cirq.bit_flip(p=bit_flip_probability).on(q) for q in qubits]

    # Measurement layer — keep in Cirq only
    meas_ops = [cirq.measure(*qubits, key="result")]

    # Full Cirq circuit for baseline simulation (with noise and measurement)
    full_circ = cirq.Circuit()
    full_circ += prep_circ
    full_circ.append(noise_ops)
    full_circ += bob_unitary_circ
    full_circ.append(meas_ops)

    if verbose:
        print("Original Cirq full circuit:")
        print(full_circ)

    # ---------------------------
    # 3) Build unitary-only circuit to send to PyZX (prep + bob_unitary)
    unitary_to_opt = cirq.Circuit()
    unitary_to_opt += prep_circ
    unitary_to_opt += bob_unitary_circ

    # Decompose to increase chance of clean QASM mapping
    decomposed_ops = []
    for op in unitary_to_opt.all_operations():
        decomposed_ops.extend(cirq.decompose(op))
    unitary_decomp = cirq.Circuit(decomposed_ops)

    # 4) Convert to Qiskit, transpile to QASM-friendly basis, export QASM
    qiskit_circ = from_cirq(unitary_decomp)
    # Use basis that Qiskit outputs to OpenQASM 2.0 comfortably
    qiskit_circ = transpile(qiskit_circ, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=0)
    qasm2 = qiskit_circ.qasm()
    if verbose:
        print("\nOpenQASM (unitary-only) sent to PyZX:")
        print(qasm2)

    # 5) Parse QASM into PyZX, convert to ZX-graph, run optimization
    pyzx_unitary = zx.Circuit.from_qasm(qasm2)
    g = pyzx_unitary.to_graph()
    zx.simplify.full_reduce(g)   # or zx.teleport_reduce(g) for T-only optimisation

    # 6) Extract optimized pyzx circuit
    c_opt = zx.extract.extract_circuit(g.copy())

    if verbose:
        print("\nPyZX optimized circuit (pyzx representation):")
        print(c_opt)

    # 7) Convert optimized pyzx circuit back to Cirq (best-effort)
    try:
        optimized_unitary_cirq = pyzx_to_cirq(c_opt, qubits)
    except NotImplementedError as e:
        # If mapping fails, export optimized QASM and inspect/patch manually
        print("Automatic pyzx->cirq mapping failed:", e)
        print("Optimized QASM (inspect for unsupported gates):\n", c_opt.to_qasm())
        raise

    if verbose:
        print("\nOptimized unitary as Cirq circuit:")
        print(optimized_unitary_cirq)

    # 8) Reconstruct full protocol: optimized unitary + noise + measurement (both in Cirq)
    new_full_circ = cirq.Circuit()
    new_full_circ += optimized_unitary_cirq
    new_full_circ.append(noise_ops)        # apply noise after unitary (same place as before)
    new_full_circ.append(meas_ops)         # measurement
    if verbose:
        print("\nReconstructed full protocol with optimized unitary:")
        print(new_full_circ)

    # 9) Simulate both original and optimized full circuits to compare
    simulator = cirq.DensityMatrixSimulator()

    # original baseline simulation
    res_orig = simulator.run(full_circ, repetitions=1)
    measured_bits_orig = res_orig.measurements["result"][0]

    # optimized simulation
    res_opt = simulator.run(new_full_circ, repetitions=1)
    measured_bits_opt = res_opt.measurements["result"][0]

    # sift according to bases
    alice_sifted = filter_qubits(states_a, encoding_basis_a, measurement_basis_b)
    bob_sifted_orig = filter_qubits(measured_bits_orig, encoding_basis_a, measurement_basis_b)
    bob_sifted_opt = filter_qubits(measured_bits_opt, encoding_basis_a, measurement_basis_b)

    if verbose:
        print("\nResults (single-shot simulations shown):")
        print(" Alice bits (original):", array_to_string(states_a))
        print(" Bob measured (orig):   ", array_to_string(measured_bits_orig))
        print(" Bob measured (opt):    ", array_to_string(measured_bits_opt))
        print(" Alice sifted:", array_to_string(alice_sifted))
        print(" Bob sifted (orig):", array_to_string(bob_sifted_orig))
        print(" Bob sifted (opt): ", array_to_string(bob_sifted_opt))

    # T-count and stats from PyZX
    t_orig = pyzx_unitary.tcount()
    t_opt = c_opt.tcount()
    if verbose:
        print(f"\nT-count (unitary): original={t_orig} optimized={t_opt}")

    # Return useful objects
    return {
        "original_full_circuit": full_circ,
        "optimized_full_circuit": new_full_circ,
        "pyzx_original_unitary": pyzx_unitary,
        "pyzx_optimized_unitary": c_opt,
        "alice_sifted": alice_sifted,
        "bob_sifted_orig": bob_sifted_orig,
        "bob_sifted_opt": bob_sifted_opt,
    }

# Example run
if __name__ == "__main__":
    out = optimize_bb84_unitary_and_run(number_of_qubits=4, bit_flip_probability=0.1, verbose=True)
