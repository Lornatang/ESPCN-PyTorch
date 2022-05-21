import numpy as np

import torch
from torch import nn
from torch.autograd import Function

from qnn.quantum_circuit import QuantumCircuit


class QuantumFunction(Function):
    @staticmethod
    def forward(ctx, batches: torch.Tensor, n_circuits, qubits, circuit, shift):
        ctx.n_circuits = n_circuits
        ctx.qubits = qubits
        ctx.circuit = circuit
        ctx.shift = shift

        # shape: (256, 20)

        # Split batch into arrays of qubit size
        split_batches = batches.unfold(1, ctx.qubits, ctx.qubits)

        # shape: (256, 10, 2)

        # Collapse first two dimensions (array of experiments)
        experiments = torch.flatten(split_batches, 0, 1)

        # shape: (2560, 2)

        z_expectations = ctx.circuit.run(experiments)

        # shape: (2560)

        result = torch.stack(z_expectations.split(ctx.n_circuits))

        # shape: (256, 10)

        ctx.save_for_backward(batches, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        batches, z_expectations = ctx.saved_tensors

        batch_size = batches.size()[0]
        parameters = ctx.n_circuits * ctx.qubits

        # shape: (256, 20)

        # We need to calculate dy/dq, that is, the derivative of the output w.r.t each qubit.
        # For that we shift input for qubit q leaving the others untouched, calculate the
        # derivative using the shift-parameter rule and repeat for all qubits.

        shifts_right = []
        shifts_left = []
        for j in range(ctx.qubits):
            # Matrix to shift only the columns corresponding to current qubit
            shift_matrix = np.zeros(batches.size())
            shift_matrix[:, list(range(j, parameters, ctx.qubits))] = ctx.shift
            shift_matrix = torch.tensor(shift_matrix)

            # Add shift to input tensor
            shift_right = batches + shift_matrix
            shift_left = batches - shift_matrix
            # Split batches into arrays of qubit size
            shift_right = shift_right.unfold(1, ctx.qubits, ctx.qubits)
            shift_left = shift_left.unfold(1, ctx.qubits, ctx.qubits)

            # shape: (256, 10, 2)

            shifts_right.append(shift_right)
            shifts_left.append(shift_left)

        experiments_right = torch.stack(shifts_right)
        experiments_left = torch.stack(shifts_left)

        # shape: (|Q|, 256, 10, 2)

        experiments = torch.stack([experiments_right, experiments_left])

        # shape: (2, |Q|, 256, 10, 2)

        experiments = torch.flatten(experiments, 0, 3)

        # shape: (2 * |Q| * 256 * 10, 2)

        results = ctx.circuit.run(experiments)

        # shape: (2 * |Q| * 256 * 10)

        results = results.view(2, ctx.qubits, batch_size, -1)

        # shape: (2, |Q|, 256, 10)

        expectations_right, expectations_left = results

        # This is our dy/dq
        gradient = expectations_right - expectations_left / 2

        # shape: (|Q|, 256, 10)

        # This is dL/dy, the derivative of the Loss w.r.t the output. This is given by torch
        # autograd we only need to convert it in order to multiply it with our gradient
        grad_output_transformed = grad_output.unsqueeze(0).repeat_interleave(ctx.qubits, 0)

        # We previously calculated dy/dq, now we multiply that by dL/dy to obtain dL/dq which is
        # the derivative of the Loss w.r.t each qubit and pass it up the backpropagation chain
        batch_gradients = gradient * grad_output_transformed

        # shape: (|Q|, 256, 10)

        # Arrange gradients to match their corresponding qubit
        out_tensor = batch_gradients.transpose(0, 1).flatten(1)

        # shape: (256, 10 * |Q|)

        return out_tensor, None, None, None, None


class QuantumLinear(nn.Module):
    def __init__(self, n_circuits, qubits, backend, shots, shift):
        super(QuantumLinear, self).__init__()
        self.n_circuits = n_circuits
        self.qubits = qubits
        # Just create one circuit as all are equal
        self.circuit = QuantumCircuit(qubits, backend, shots)
        self.shift = shift

    def forward(self, batches):
        return QuantumFunction.apply(batches, self.n_circuits, self.qubits, self.circuit, self.shift)
