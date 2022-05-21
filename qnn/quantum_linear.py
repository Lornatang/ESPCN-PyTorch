import numpy as np

import torch
from torch import nn
from torch.autograd import Function

from qnn.quantum_circuit import QuantumCircuit
from qnn.thread_functions import run_forward_batch, run_backward_batch


class QuantumFunction(Function):
    @staticmethod
    def forward(ctx, batches: torch.Tensor, circuits, shift, pool):
        ctx.shift = shift
        ctx.circuits = circuits
        ctx.pool = pool

        # Split batch into arrays of qubit size
        split_batches = [np.split(batch, len(circuits)) for batch in batches.numpy()]

        # Run batches and wait for result
        if ctx.pool:
            z_expectations = ctx.pool.starmap(run_forward_batch, [(batch, circuits) for batch in split_batches])
        else:
            z_expectations = [run_forward_batch(batch, circuits) for batch in split_batches]

        result = torch.tensor(np.array(z_expectations))
        ctx.save_for_backward(batches, result)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        batches, expectation_z = ctx.saved_tensors
        input_list = batches.numpy()

        parameters = input_list.shape[1]
        qubits = parameters // len(ctx.circuits)
        results = [[None for _ in range(qubits)] for _ in range(len(input_list))]

        # TODO: optimize derivative w.r.t each qubit somehow
        # Derivative of the output w.r.t each qubit dy/dq.
        # Shift one qubit and leave the others untouched, repeat for all.
        for j in range(qubits):
            # Matrix to shift only the columns corresponding to current qubit
            shift_matrix = np.zeros(input_list.shape)
            shift_matrix[:, list(range(j, parameters, qubits))] = ctx.shift

            # Add shift to input tensor
            shift_right = input_list + shift_matrix
            shift_left = input_list - shift_matrix
            # Split batch into arrays of qubit size
            shifts_right = [np.split(shift, len(ctx.circuits)) for shift in shift_right]
            shifts_left = [np.split(shift, len(ctx.circuits)) for shift in shift_left]

            # Start execution and save async result to be retrieved later
            for i, batch in enumerate(zip(shifts_right, shifts_left)):
                if ctx.pool:
                    results[i][j] = ctx.pool.apply_async(run_backward_batch, (batch, ctx.circuits))
                else:
                    results[i][j] = run_backward_batch(batch, ctx.circuits)

        # Wait for all tasks
        if ctx.pool:
            results = torch.tensor([[v.get() for v in r] for r in results])
        else:
            results = torch.tensor(results)

        # This is dL/dy, the derivative of the Loss w.r.t the output. This is given by torch
        # autograd we only need to convert it in order to multiply it with our gradient
        grad_output_transformed = grad_output.unsqueeze(1).repeat_interleave(qubits, 1)

        # We previously calculated dy/dq, now we multiply that by dL/dy to obtain dL/dq which is
        # the derivative of the Loss w.r.t each qubit and pass it up the backpropagation chain
        batch_gradients = results * grad_output_transformed

        # Arrange gradients to match their corresponding qubit
        out_tensor = batch_gradients.transpose(1, 2).flatten(1)

        return out_tensor.float(), None, None, None


class QuantumLinear(nn.Module):
    def __init__(self, n_circuits, qubits, backend, shots, shift, pool):
        super(QuantumLinear, self).__init__()
        self.circuits = [QuantumCircuit(qubits, backend, shots) for _ in range(n_circuits)]
        self.shift = shift
        self.pool = pool

    def forward(self, batches):
        return QuantumFunction.apply(batches, self.circuits, self.shift, self.pool)
