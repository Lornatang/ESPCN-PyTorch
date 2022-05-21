import qiskit
from qiskit import transpile, assemble


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(n_qubits)

        all_qubits = [i for i in range(n_qubits)]
        self.thetas = qiskit.circuit.ParameterVector('thetas', n_qubits)

        self._circuit.h(all_qubits)
        self._circuit.barrier()
        for i, theta in enumerate(self.thetas):
            self._circuit.ry(theta, i)

        self._circuit.measure_all()
        # ---------------------------

        self.backend = backend
        self.shots = shots
        # Transpile & assemble circuit in advance to speed-up execution
        self.t_qc = transpile(self._circuit, self.backend)
        self.qobj = assemble(
            self.t_qc,
            shots=self.shots,
            parameter_binds=[{theta: 0 for theta in self.thetas}]
        )

    def run(self, thetas):
        for i, theta in enumerate(thetas):
            # Modify params directly into the instruction list to avoid re-assembling the circuit
            # TODO: investigate if Qiskit has a method to do this on a qObj.
            self.qobj.experiments[0].instructions[1 + len(thetas) + i].params[0] = qiskit.circuit.ParameterExpression({}, theta)

        job = self.backend.run(self.qobj)
        counts = job.result().get_counts()

        # Get pauli-Z expectation
        val = 0.0
        for bitstring, count in counts.items():
            sign = (-1) ** bitstring.count("0")
            val += sign * count

        return val / self.shots