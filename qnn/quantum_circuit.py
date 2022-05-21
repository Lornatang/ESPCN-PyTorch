import copy
import qiskit
import torch
from qiskit import transpile, assemble


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        self.n_qubits = n_qubits

        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)

        all_qubits = [i for i in range(self.n_qubits)]
        self.thetas = qiskit.circuit.ParameterVector('thetas', self.n_qubits)

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

    def run(self, experiments):
        device = experiments.device
        experiments = experiments.cpu().detach().numpy()

        qobj = assemble(
            self.t_qc,
            shots=self.shots,
            parameter_binds=[{theta: 0 for theta in self.thetas}]
        )
        qobj.experiments = QuantumCircuit.get_experiments(qobj.experiments[0], experiments)

        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts()

        expectations = []
        for exp in counts:
            # Get pauli-Z expectation
            val = 0.0
            for bitstring, count in exp.items():
                sign = (-1) ** bitstring.count("0")
                val += sign * count
            expectations.append(val)

        expectations = torch.tensor(expectations).to(device)

        return expectations / self.shots

    @staticmethod
    def get_experiments(qobj_exp, experiments):
        # Modify params directly into the instruction list to avoid re-assembling the circuit
        # TODO: investigate if Qiskit has a method to do this better.
        qobj_exps = []
        first = next(i for i, ins in enumerate(qobj_exp.instructions) if ins.name == 'ry')
        for thetas in experiments:
            shallow = copy.copy(qobj_exp)
            shallow.instructions = shallow.instructions[:]
            for i, theta in enumerate(thetas):
                shallow.instructions[first + i] = copy.copy(shallow.instructions[first + i])
                shallow.instructions[first + i].params = shallow.instructions[first + i].params[:]
                shallow.instructions[first + i].params[0] = qiskit.circuit.ParameterExpression({}, theta)
            qobj_exps.append(shallow)
        return qobj_exps
