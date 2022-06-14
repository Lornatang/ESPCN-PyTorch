import torch
from torch import nn

from qcnn.qcnn_kernel import QuantumKernel

class QuantumLinear(nn.Module):
    def __init__(self, n_circuits, qubits, backend, shots, shift):
        super(QuantumLinear, self).__init__()
        self.n_circuits = n_circuits
        self.qubits = qubits

        # Create circuits
        self.circuits = nn.ModuleList([
            QuantumKernel((1, self.qubits), backend, shots, shift)
            for _ in range(self.n_circuits)
        ])

    def forward(self, batches):
        n, m = batches.size()

        # q = qubits
        # c = circuits
        # m = q * c

        # shape: (n, m)

        split_batches = batches.unfold(1, self.qubits, self.qubits)

        # shape: (n, c, q)

        # Convert to batches for each circuit

        circuit_batches = torch.transpose(split_batches, 0, 1)

        # shape: (c, n, q)

        batch_results = [self.circuits[i](batch) for i, batch in enumerate(circuit_batches)]

        # shape: [c](n)

        batch_results = torch.stack(batch_results)

        # shape: (c, n)

        results = batch_results.transpose(0, 1)

        # shape: (n, c)

        return results
