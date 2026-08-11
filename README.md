# BB84 + Golay Key Distillation (Cirq port)

A Cirq re-implementation of the AWS `sample-BB84-qkd-on-amazon-braket`
project: BB84 quantum key distribution with simulated channel noise and
`[24,12,8]` Golay-code error correction for key reconciliation.

## Project structure

```
bb84-cirq-golay/
├── main.py                  # entry point, runs the full pipeline
├── requirements.txt
└── utils/
    ├── bb84_cirq.py          # Cirq circuits: encode, measure, sift
    ├── golay_code.py         # Golay generator / parity-check / B matrices
    ├── error_correction.py   # Alice/Bob reconciliation steps
    └── secret_utils.py       # bit-array -> bytes/base64
```

## Setup (VS Code)

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. In VS Code, select this venv as your Python interpreter
   (`Ctrl+Shift+P` → "Python: Select Interpreter" → pick `./venv`).
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run

```
python main.py
```

You'll see Alice's and Bob's raw sifted keys (which will usually differ
in 1-2 bits due to the simulated 10% bit-flip channel noise), the parity
bits Alice sends over the classical channel, and Bob's corrected key —
which should end up identical to Alice's as long as 3 or fewer bits
differed.

## Configuration

Edit the constants at the top of `main.py`:

| Parameter | Meaning | Default |
|---|---|---|
| `BIT_FLIP_PROBABILITY` | Simulated channel noise per qubit | `0.1` |
| `NUMBER_OF_QUBITS` | Qubits sent per BB84 round | `12` |
| `ERROR_CORRECTION_CHUNK_SIZE` | Sifted bits needed before Golay correction | `12` |

Keep `ERROR_CORRECTION_CHUNK_SIZE` at `12` unless you also change the
Golay code (it's hard-coded to `[24,12,8]`).

## How it works

1. **`run_round()`** (in `bb84_cirq.py`) builds a Cirq circuit: Alice
   preps qubits with X/H gates for her random bits/bases, `cirq.bit_flip`
   noise is applied to model channel imperfections, Bob applies his own
   H gates and measures via `cirq.DensityMatrixSimulator`.
2. **Sifting** keeps only bits where Alice's and Bob's bases matched.
3. Rounds repeat until 12 sifted bits accumulate.
4. **Reconciliation**: Alice encodes her 12-bit key into a 24-bit Golay
   codeword and sends only the 12 parity bits. Bob combines those with
   his own raw key, computes a syndrome, and (if 3 or fewer bits
   differed) recovers the exact error pattern and corrects his key —
   without either side ever transmitting the actual key bits.

## Notes

- Uses `cirq.DensityMatrixSimulator`, Cirq's equivalent of Braket's
  `braket_dm` — needed because `cirq.bit_flip` is a noise channel, not
  a unitary gate, and only density-matrix (or trajectory) simulators
  can model it.
- The Golay math was independently re-derived and numerically verified
  (self-dual, min distance 8, 759 minimum-weight codewords) rather than
  copied from any existing implementation.
