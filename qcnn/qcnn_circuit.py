import ctypes
import math
import torch
import numpy as np
import pathlib

directory = pathlib.Path(__file__).parent.resolve().__str__()
# clib = ctypes.cdll.LoadLibrary(directory + "/bridge/libangle.so")
clib = ctypes.cdll.LoadLibrary(directory + "/bridge/libamplitude.so")

clib.run_inputs.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

clib.run_thetas.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]


class QuantumCircuit:
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.n_feature_qubits = math.ceil(math.log2(n_inputs + 1))
        self.n_thetas = self.n_feature_qubits * 3

    def run_inputs(self, inputs, thetas):
        device = inputs.device

        inputs = inputs.cpu().numpy().astype(np.float32)
        thetas = thetas.cpu().numpy().astype(np.float32)
        expectations = np.empty(shape=inputs.shape[0], dtype=np.float32)

        clib.run_inputs(expectations, inputs, thetas, inputs.shape[0], inputs.shape[1], 0)

        return torch.tensor(expectations).to(device)

    def run_thetas(self, inputs, thetas):
        device = inputs.device

        inputs = inputs.cpu().numpy().astype(np.float32)
        thetas = thetas.cpu().numpy().astype(np.float32)
        expectations = np.empty(shape=inputs.shape[0] * thetas.shape[0], dtype=np.float32)

        clib.run_thetas(expectations, inputs, thetas, inputs.shape[0], inputs.shape[1], thetas.shape[0], 0)

        return torch.tensor(expectations).to(device)





