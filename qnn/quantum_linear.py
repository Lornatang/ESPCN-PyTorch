import numpy as np

import torch
from torch import nn
from torch.autograd import Function

from qnn.quantum_circuit import QuantumCircuit


class QuantumFunction(Function):
    @staticmethod
    def forward(ctx, batches: torch.Tensor, thetas: torch.Tensor, n_circuits, circuit, shift):
        ctx.n_circuits = n_circuits
        ctx.qubits = circuit.n_qubits
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.batch_size = batches.size()[0]

        # shape: (256, 20)

        # Split batch into arrays of qubit size
        split_batches = batches.unfold(1, ctx.qubits, ctx.qubits)

        # shape: (256, 10, 2)

        # Collapse first two dimensions (array of experiments)
        experiments = torch.flatten(split_batches, 0, 1)

        # shape: (2560, 2)

        # Repeat thetas to have one per batch
        transformed_thetas = thetas.repeat(ctx.batch_size, 1)
        # Run the circuit
        z_expectations = ctx.circuit.run(experiments, transformed_thetas)

        # shape: (2560)

        result = torch.stack(z_expectations.split(ctx.n_circuits))

        # shape: (256, 10)

        ctx.save_for_backward(batches, thetas, result)

        return result

    @staticmethod
    def get_shifted_param(ctx, param: torch.Tensor):
        t, _, _ = ctx.saved_tensors
        device = t.device
        n_parameters = ctx.n_circuits * ctx.qubits

        # shape: (256, 20)

        # We need to calculate dy/dq, that is, the derivative of the output w.r.t each param
        # (qubit/theta). For that we shift input for one param leaving the others untouched,
        # calculate the derivative using the shift-parameter rule and repeat for all params.

        shifts_right = []
        shifts_left = []
        for j in range(ctx.qubits):
            # Matrix to shift only the columns corresponding to current param
            shift_matrix = np.zeros(param.size())
            shift_matrix[:, list(range(j, n_parameters, ctx.qubits))] = ctx.shift
            shift_matrix = torch.tensor(shift_matrix).to(device)

            # Add shift to input tensor
            shift_right = param + shift_matrix
            shift_left = param - shift_matrix
            # Split batches into arrays of qubit size
            shift_right = shift_right.unfold(1, ctx.qubits, ctx.qubits)
            shift_left = shift_left.unfold(1, ctx.qubits, ctx.qubits)

            # shape: (256, 10, 2)

            shifts_right.append(shift_right)
            shifts_left.append(shift_left)

        shifted_param_right = torch.stack(shifts_right)
        shifted_param_left = torch.stack(shifts_left)

        # shape: (|Q|, 256, 10, 2)

        shifted_param = torch.stack([shifted_param_right, shifted_param_left])

        # shape: (2, |Q|, 256, 10, 2)

        shifted_param = torch.flatten(shifted_param, 0, 3)

        # shape: (2 * |Q| * 256 * 10, 2)

        return shifted_param

    @staticmethod
    def get_result_gradient(ctx, grad_output, result: torch.Tensor):

        # shape: (2 * |Q| * 256 * 10)

        result = result.view(2, ctx.qubits, ctx.batch_size, -1)

        # shape: (2, |Q|, 256, 10)

        expectations_right, expectations_left = result

        # This is our dy/dq
        gradient = (expectations_right - expectations_left) / 2

        # shape: (|Q|, 256, 10)

        # This is dL/dy, the derivative of the Loss w.r.t the output. This is given by torch
        # autograd we only need to convert it in order to multiply it with our gradient
        grad_output_transformed = grad_output.unsqueeze(0).repeat_interleave(ctx.qubits, 0)

        # We previously calculated dy/dq, now we multiply that by dL/dy to obtain dL/dq which is
        # the derivative of the Loss w.r.t each qubit/theta and pass it up the backpropagation chain
        batch_gradients = gradient * grad_output_transformed

        # shape: (|Q|, 256, 10)

        # Arrange gradients to match their corresponding qubit/theta
        out_tensor = batch_gradients.transpose(0, 1).transpose(1, 2).reshape(ctx.batch_size, ctx.n_circuits, -1)

        # shape: (256, 10 * |Q|)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        batches, thetas, z_expectations = ctx.saved_tensors

        # To understand all these transformations you should check input/output
        # shapes of the shift, run and gradient functions. Basically what the
        # transformations do is to convert batch and theta tensors into adequate
        # format to reuse the same logic for shifting, running the circuit
        # and calculating the gradient

        # Transform non-shifted inputs to be passed together with shifted thetas
        normal_inputs = batches.unfold(1, ctx.qubits, ctx.qubits)
        normal_inputs = torch.flatten(normal_inputs, 0, 1)
        normal_inputs = normal_inputs.repeat(2 * ctx.qubits, 1)

        # Transform non-shifted thetas to be passed together with shifted inputs
        normal_thetas = thetas.repeat(2 * ctx.qubits * ctx.batch_size, 1)

        # Transform thetas to match same input format as batches (just to reuse same logic)
        thetas = thetas.unsqueeze(0).repeat_interleave(ctx.batch_size, 0)
        thetas = thetas.flatten(1)
        # Shift inputs and thetas to be passed together with non-shifted pairs
        shifted_inputs = QuantumFunction.get_shifted_param(ctx, batches)
        shifted_thetas = QuantumFunction.get_shifted_param(ctx, thetas)

        # Do the pass. Run all batches for each shift and collect results
        inputs_result = ctx.circuit.run(shifted_inputs, normal_thetas)
        thetas_result = ctx.circuit.run(normal_inputs, shifted_thetas)

        # Get gradient for the obtained results
        inputs_gradient = QuantumFunction.get_result_gradient(ctx, grad_output, inputs_result)
        thetas_gradient = QuantumFunction.get_result_gradient(ctx, grad_output, thetas_result)

        # Do some transforms to match input format in forward method
        inputs_gradient = inputs_gradient.flatten(1)
        thetas_gradient = thetas_gradient.mean(0)

        return inputs_gradient, thetas_gradient, None, None, None


class QuantumLinear(nn.Module):
    def __init__(self, n_circuits, qubits, backend, shots, shift):
        super(QuantumLinear, self).__init__()
        self.n_circuits = n_circuits
        self.qubits = qubits
        self.shift = shift
        # Just create one circuit as all are equal
        self.circuit = QuantumCircuit(qubits, backend, shots)
        # Initialize thetas (weights)
        self.thetas = torch.nn.Parameter(torch.FloatTensor(n_circuits, qubits).uniform_(-torch.pi/2, torch.pi/2))

    def forward(self, batches):
        return QuantumFunction.apply(batches, self.thetas, self.n_circuits, self.circuit, self.shift)
