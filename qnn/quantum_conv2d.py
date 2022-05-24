import torch
from torch import nn
import torch.nn.functional as F

from qnn.quantum_linear import QuantumLinear

class QuantumConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, backend, shots, shift, stride=1):
        super(QuantumConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        # TODO: implement later
        # self.stride = stride

        # Create circuits
        kH, kW = self.kernel_size
        self.circuits = nn.ModuleList([
            QuantumLinear(kH, kW, backend, shots, shift)
            for _ in range(self.out_channels * self.in_channels * kH)
        ])

    def forward(self, batches):
        n, iC, H, W = batches.size()
        oC, iC, kH, kW, pH, pW = self.out_channels, self.in_channels, *self.kernel_size, *self.padding
        oH, oW = H + 2 * pH - (kH - 1), W + 2 * pW - (kW - 1)

        # shape: (n, iC, H, W)
        in_channels = batches.transpose(0, 1)
        # shape: (iC, n, H, W)
        # Add padding
        padded_in_channels = F.pad(in_channels, (pH, pH, pW, pW))
        # shape: (iC, n, H + pH, W + pW)

        out_channels = []  # shape: [oC](n, oH, oW)
        for i in range(oC):
            outputs = []  # shape: [iC](n, oH, oW)

            for j, channel in enumerate(padded_in_channels):
                # shape: (n, H + pH, W + pW)
                # Extract the (kH, kW) inputs that will be applied to the kernel
                inputs = [channel[:, i: i + kH, j: j + kW] for i in range(oH) for j in range(oW)]
                inputs = torch.stack(inputs, 1)
                # shape: (n, oH * oW, kH, kW)
                # Convert to batches of qubits
                linear_batches = inputs.flatten(0, 1).flatten(1)
                # shape: (n * oH * oW, kH * kW)
                linear_result = self.circuits[i * j](linear_batches)
                # shape: (n * oH * oW, kH)
                linear_result = linear_result.view(n, oH, oW, kH)
                # shape: (n, oH, oW, kH)
                # Average the results for each row of the kernel
                average = torch.mean(linear_result, dim=3)
                # shape: (n, oH, oW)
                outputs.append(average)

            # Aggregate outputs into a single out channel
            out_channel = torch.sum(torch.stack(outputs), dim=0)
            out_channels.append(out_channel)

        # Obtain final output
        result = torch.stack(out_channels)
        # shape: (oC, n, oH, oW)
        result = result.transpose(0, 1)
        # shape: (n, oC, oH, oW)

        return result
