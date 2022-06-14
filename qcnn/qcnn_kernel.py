import math
import numpy as np
import torch

from torch import nn
from torch.autograd import Function

from qcnn.qcnn_circuit import QuantumCircuit


class QuantumKernelFunction(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, thetas: torch.Tensor, circuit: QuantumCircuit, shift: int):
        ctx.qubits = circuit.n_feature_qubits
        ctx.circuit = circuit
        ctx.shift = shift
        ctx.n_inputs = inputs.shape[1]
        ctx.input_size = inputs.shape[0]

        # N = n * oH * oW
        # n = kH * kW
        # shape: (N, n)

        # Run the circuit
        experiments = ctx.circuit.build_input_experiments(inputs, thetas)
        z_expectations = ctx.circuit.run(experiments, inputs.device)

        # shape: (N)

        ctx.save_for_backward(inputs, thetas, z_expectations)

        return z_expectations

    @staticmethod
    def get_shifted_param(ctx, param: torch.Tensor):
        t, _, _ = ctx.saved_tensors
        device = t.device
        n = param.shape[1]

        # shape: (N, n)

        # We need to calculate dy/dq, that is, the derivative of the output w.r.t each param
        # (input/theta). For that we shift one param leaving the others untouched, calculate
        # the derivative using the shift-parameter rule and repeat for all params.

        shifts_right = []
        shifts_left = []
        for j in range(n):
            # Matrix to shift only the columns corresponding to current param
            shift_matrix = np.zeros(param.size())
            shift_matrix[:, list(range(j, n, n))] = ctx.shift
            shift_matrix = torch.tensor(shift_matrix).to(device)

            # Add shift to input tensor
            shift_right = param + shift_matrix
            shift_left = param - shift_matrix

            shifts_right.append(shift_right)
            shifts_left.append(shift_left)

        shifted_param_right = torch.stack(shifts_right)
        shifted_param_left = torch.stack(shifts_left)

        # shape: (n, N, n)

        shifted_param = torch.stack([shifted_param_right, shifted_param_left])

        # shape: (2, n, N, n)

        shifted_param = torch.flatten(shifted_param, 0, 2)

        # shape: (2 * n * N, n)

        return shifted_param

    @staticmethod
    def get_result_gradient(ctx, grad_output, result: torch.Tensor, n: int):

        # shape: (2 * n * N)

        result = result.view(2, n, ctx.input_size)

        # shape: (2, n, N)

        expectations_right, expectations_left = result

        # This is our dy/dq
        gradient = (expectations_right - expectations_left) / (2 * math.sin(ctx.shift))

        # shape: (n, N)

        # This is dL/dy, the derivative of the Loss w.r.t the output. This is given by torch
        # autograd we only need to convert it in order to multiply it with our gradient
        grad_output_transformed = grad_output.unsqueeze(0).repeat_interleave(n, 0)

        # We previously calculated dy/dq, now we multiply that by dL/dy to obtain dL/dq which is
        # the derivative of the Loss w.r.t each qubit/theta and pass it up the backpropagation chain
        input_gradients = gradient * grad_output_transformed

        # shape: (n, N)

        # Arrange gradients to match their corresponding qubit/theta
        out_tensor = input_gradients.transpose(0, 1)

        # shape: (N, n)

        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        inputs, thetas, z_expectations = ctx.saved_tensors

        # shape: (N, n)

        transformed_thetas = torch.unsqueeze(thetas, 0)  # Shape it like the input

        # Shift inputs and thetas to be passed together with non-shifted pairs
        shifted_inputs = QuantumKernelFunction.get_shifted_param(ctx, inputs)
        shifted_thetas = QuantumKernelFunction.get_shifted_param(ctx, transformed_thetas)

        # Do the pass. Run all inputs for each shift and collect results
        input_experiments = ctx.circuit.build_input_experiments(shifted_inputs, thetas)
        theta_experiments = ctx.circuit.build_theta_experiments(inputs, shifted_thetas)

        results = ctx.circuit.run([*input_experiments, *theta_experiments], inputs.device)

        # Separate results
        input_results = results[:len(input_experiments)]
        theta_results = results[len(input_experiments):]

        # Get gradient for the obtained results
        inputs_gradient = QuantumKernelFunction.get_result_gradient(ctx, grad_output, input_results, inputs.shape[1])
        thetas_gradient = QuantumKernelFunction.get_result_gradient(ctx, grad_output, theta_results, thetas.shape[0])

        return inputs_gradient, thetas_gradient, None, None, None


class QuantumKernel(nn.Module):
    def __init__(self, kernel_size, backend, shots, shift):
        super(QuantumKernel, self).__init__()
        self.kernel_size = kernel_size
        self.shift = shift
        self.circuit = QuantumCircuit(kernel_size, backend, shots)
        # Initialize thetas (weights)
        self.thetas = torch.nn.Parameter(torch.FloatTensor(self.circuit.n_thetas).uniform_(-torch.pi/2, torch.pi/2))

    def forward(self, inputs):
        return QuantumKernelFunction.apply(inputs, self.thetas, self.circuit, self.shift)
