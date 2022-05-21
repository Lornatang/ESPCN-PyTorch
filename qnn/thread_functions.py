from typing import Tuple

import numpy as np


def run_forward_batch(batch: np.ndarray, circuits):
    return [circuit.run(input.tolist()) for input, circuit in zip(batch, circuits)]


def run_backward_batch(batch: Tuple[np.ndarray, np.ndarray], circuits):
    r, l = batch
    expectations_right = [circuit.run(shift) for shift, circuit in zip(r, circuits)]
    expectations_left = [circuit.run(shift) for shift, circuit in zip(l, circuits)]
    return [(r - l) / 2 for r, l in zip(expectations_right, expectations_left)]
