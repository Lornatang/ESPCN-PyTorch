import copy
import numpy as np
import qiskit
from qiskit.result.models import ExperimentResult

def get_results(qobj, backend, experiments: list) -> list:
    qobj = copy.copy(qobj)
    qobj.experiments = experiments

    job = backend.run(qobj)
    result = job.result()

    # Qiskit get_counts is too slow, extract HEX results from the result object
    # and use them to count the amount of zeroes which is what we need to
    # calculate the expectation.
    return [get_expectation(result) for result in result.results]


def get_expectation(result: ExperimentResult):
    counts_dict = result.data.counts

    # Get pauli-Z expectation
    signs = np.array([-1 if get_zeroes(h) % 2 else 1 for h in counts_dict.keys()])
    counts = np.array(list(counts_dict.values()))
    values = signs * counts

    return np.sum(values)


def get_zeroes(hex_: str) -> int:
    return bin(int(hex_, 16))[2:].count('0')


def get_input_experiments(qobj_exp, inputs: np.ndarray, thetas: np.ndarray, qubits: int):
    # Modify params directly into the instruction list to avoid re-assembling the circuit
    amplitudes = encode_to_amplitudes(inputs, qubits)
    experiments = []

    n_thetas = thetas.shape[0]
    n_thetas_per_cu3 = 3
    n_cu3s = n_thetas // n_thetas_per_cu3

    # Build parameter instructions
    first_cu3 = next(i for i, ins in enumerate(qobj_exp.instructions) if ins.name == 'cu3')
    cu3_instructions = qobj_exp.instructions[first_cu3:first_cu3 + n_cu3s]
    for i in range(n_cu3s):
        for j in range(n_thetas_per_cu3):
            cu3_instructions[i] = copy.copy(cu3_instructions[i])
            cu3_instructions[i].params = cu3_instructions[i].params[:]
            cu3_instructions[i].params[j] = qiskit.circuit.ParameterExpression({}, thetas[i + j])

    # Bind inputs & set reference to parameters
    ix_input = next(i for i, ins in enumerate(qobj_exp.instructions) if ins.name == 'initialize')
    for xs in amplitudes:
        shallow = copy.copy(qobj_exp)
        shallow.instructions = shallow.instructions[:]
        shallow.instructions[ix_input] = copy.copy(shallow.instructions[ix_input])
        shallow.instructions[ix_input].params = [complex(x, 0) for x in xs]
        shallow.instructions[first_cu3:first_cu3 + n_thetas] = cu3_instructions
        experiments.append(shallow)
    return experiments


def prebuild_theta_inputs(qobj_exp, inputs: np.ndarray, qubits: int) -> list:
    amplitudes = encode_to_amplitudes(inputs, qubits)

    # Build input instructions
    ix_input = next(i for i, ins in enumerate(qobj_exp.instructions) if ins.name == 'initialize')
    input_instructions = []
    for xs in amplitudes:
        input_instruction = copy.copy(qobj_exp.instructions[ix_input])
        input_instruction.params = [complex(x, 0) for x in xs]
        input_instructions.append(input_instruction)
    return input_instructions


def get_theta_experiments(qobj_exp, thetas_array: np.ndarray, input_instructions: list) -> list:
    # Modify params directly into the instruction list to avoid re-assembling the circuit
    ix_input = next(i for i, ins in enumerate(qobj_exp.instructions) if ins.name == 'initialize')
    experiments = []

    n_thetas = thetas_array.shape[1]
    n_thetas_per_cu3 = 3
    n_cu3s = n_thetas // n_thetas_per_cu3

    # Bind parameters & set reference to inputs
    first_cu3 = next(i for i, ins in enumerate(qobj_exp.instructions) if ins.name == 'cu3')
    for thetas in thetas_array:
        for input_instruction in input_instructions:
            shallow = copy.copy(qobj_exp)
            shallow.instructions = shallow.instructions[:]
            for i in range(n_cu3s):
                cu3 = first_cu3 + i
                shallow.instructions[cu3] = copy.copy(shallow.instructions[cu3])
                shallow.instructions[cu3].params = [
                    qiskit.circuit.ParameterExpression({}, thetas[n_thetas_per_cu3 * i + j])
                    for j in range(n_thetas_per_cu3)
                ]
                shallow.instructions[ix_input] = input_instruction
            experiments.append(shallow)
    return experiments


def encode_to_amplitudes(experiments: np.ndarray, qubits: int):
    # Add normalization element
    experiments = np.c_[experiments, np.ones(experiments.shape[0])]
    # Fill up to power of 2
    for _ in range(2 ** qubits - experiments.shape[1]):
        experiments = np.c_[experiments, np.zeros(experiments.shape[0])]
    # Get magnitude of each row and normalize
    squares = np.square(experiments)
    sums = np.sum(squares, axis=1)
    magnitudes = np.sqrt(sums)
    magnitudes_array = np.tile(magnitudes, (experiments.shape[1], 1)).transpose()
    normalized = experiments / magnitudes_array
    return np.flip(normalized, axis=1)  # Reverse due to qiskit qubit order