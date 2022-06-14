import math
import qiskit
import torch
import signal
import numpy as np
from multiprocessing import Pool
from qiskit import transpile, assemble
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qcnn.qcnn_circuit_helpers import get_results, get_input_experiments, get_theta_experiments, prebuild_theta_inputs


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


pool = Pool(initializer=init_worker)


class QuantumCircuit:
    def __init__(self, kernel_size, backend, shots):
        self.kernel_size = kernel_size

        # --- Circuit definition ---
        kH, kW = self.kernel_size
        # Last amplitudes will be wasted as feature vector has to be power of 2
        self.n_feature_qubits = math.ceil(math.log2(kH * kW + 1)) # Add +1 normalization element
        self.n_thetas = self.n_feature_qubits * 3
        self._circuit = RawFeatureVector(2 ** self.n_feature_qubits)
        self._circuit = self._circuit.assign_parameters(np.ones(2 ** self.n_feature_qubits))

        self.thetas = qiskit.circuit.ParameterVector('Θ', self.n_thetas)

        self._circuit.measure_all()
        self._circuit.add_register(qiskit.circuit.QuantumRegister(1, 't'))  # Target qubit
        self._circuit.barrier()
        for i in range(self.n_feature_qubits):
            self._circuit.cu3(
                self.thetas[3 * i],
                self.thetas[3 * i + 1],
                self.thetas[3 * i + 2],
                i, self.n_feature_qubits
            )
        self._circuit.barrier()

        self._circuit.draw(output='mpl', filename='circuit.png')
        # ---------------------------

        self.backend = backend
        self.shots = shots
        # Transpile and assemble circuit in advance to speed-up execution
        self.t_qc = transpile(self._circuit, self.backend)
        self.qobj = assemble(
            self.t_qc,
            shots=self.shots,
            parameter_binds=[{param: 0 for params in zip(self.thetas) for param in params}]
        )

    def build_input_experiments(self, inputs: torch.Tensor, thetas: torch.Tensor):
        inputs = inputs.cpu().detach().numpy()
        thetas = thetas.cpu().detach().numpy()

        chunks = np.array_split(inputs, pool._processes)

        chunked_experiments = pool.starmap(get_input_experiments, [(
            self.qobj.experiments[0], chunk, thetas, self.n_feature_qubits
        ) for chunk in chunks])

        return [e for c in chunked_experiments for e in c]

    def build_theta_experiments(self, inputs: torch.Tensor, thetas: torch.Tensor):
        inputs = inputs.cpu().detach().numpy()
        thetas = thetas.cpu().detach().numpy()

        input_instructions = prebuild_theta_inputs(self.qobj.experiments[0], inputs, self.n_feature_qubits)
        experiments = get_theta_experiments(self.qobj.experiments[0], thetas, input_instructions)

        return experiments

    def run(self, experiments, device):
        chunks = np.array_split(np.array(experiments), pool._processes)
        chunked_expectations = pool.starmap(get_results, [(
            self.qobj, self.backend, chunk
        ) for chunk in chunks])

        expectations = torch.tensor([e for ce in chunked_expectations for e in ce]).to(device)

        return expectations / self.shots





