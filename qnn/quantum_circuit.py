import copy
import qiskit
import torch
import numpy as np
from qiskit import transpile, assemble
from qiskit.result.models import ExperimentResult


class QuantumCircuit:
    def __init__(self, n_qubits, backend, shots):
        self.n_qubits = n_qubits

        # --- Circuit definition ---
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)

        self.inputs = qiskit.circuit.ParameterVector('x', self.n_qubits)
        self.thetas = qiskit.circuit.ParameterVector('Θ', self.n_qubits)

        for i, [x, theta] in enumerate(zip(self.inputs, self.thetas)):
            self._circuit.h(i)
            self._circuit.ry(x, i)
            self._circuit.barrier(i)
            self._circuit.rx(theta, i)

        self._circuit.measure_all()
        # self._circuit.draw(output='mpl', filename='circuit.png')
        # ---------------------------

        self.backend = backend
        self.shots = shots
        # Transpile circuit in advance to speed-up execution
        self.t_qc = transpile(self._circuit, self.backend)

    def run(self, experiments, thetas):
        device = experiments.device
        experiments = experiments.cpu().detach().numpy()
        thetas = thetas.cpu().detach().numpy()

        qobj = assemble(
            self.t_qc,
            shots=self.shots,
            parameter_binds=[{param: 0 for params in zip(self.inputs, self.thetas) for param in params}]
        )
        qobj.experiments = QuantumCircuit.get_experiments(qobj.experiments[0], experiments, thetas)

        job = self.backend.run(qobj)
        result = job.result()

        # Qiskit get_counts is too slow, extract HEX results from the result object
        # and use them to count the amount of zeroes which is what we need to
        # calculate the expectation.
        expectations = [QuantumCircuit.get_expectation(res) for res in result.results]

        expectations = torch.tensor(expectations).to(device)

        return expectations / self.shots

    @staticmethod
    def get_expectation(result: ExperimentResult):
        counts_dict = result.data.counts

        # Get pauli-Z expectation
        signs = np.array([-1 if QuantumCircuit.get_zeroes(h) % 2 else 1 for h in counts_dict.keys()])
        counts = np.array(list(counts_dict.values()))
        values = signs * counts

        return np.sum(values)

    @staticmethod
    def get_zeroes(hex_: str) -> int:
        return bin(int(hex_, 16))[2:].count('0')

    @staticmethod
    def get_experiments(qobj_exp, experiments, exp_thetas):
        # Modify params directly into the instruction list to avoid re-assembling the circuit
        # TODO: investigate if Qiskit has a method to do this better.
        qobj_exps = []
        first = next(i for i, ins in enumerate(qobj_exp.instructions) if ins.name == 'ry')
        for xs, thetas in zip(experiments, exp_thetas):
            shallow = copy.copy(qobj_exp)
            shallow.instructions = shallow.instructions[:]
            for i, [x, theta] in enumerate(zip(xs, thetas)):
                rx = first + i * 4
                ry = rx + 2
                shallow.instructions[rx] = copy.copy(shallow.instructions[rx])
                shallow.instructions[rx].params = shallow.instructions[rx].params[:]
                shallow.instructions[rx].params[0] = qiskit.circuit.ParameterExpression({}, x)
                shallow.instructions[ry] = copy.copy(shallow.instructions[ry])
                shallow.instructions[ry].params = shallow.instructions[ry].params[:]
                shallow.instructions[ry].params[0] = qiskit.circuit.ParameterExpression({}, theta)
            qobj_exps.append(shallow)
        return qobj_exps
