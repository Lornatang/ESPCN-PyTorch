#include "qfactory.hpp"
#include "qneuron.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>

#define CORES 7

using namespace Qrack;

int intPow(int x, unsigned int p) {
    if (p == 0) return 1;
    if (p == 1) return x;
    int tmp = intPow(x, p/2);
    if (p%2 == 0) return tmp * tmp;
    else return x * tmp * tmp;
}

real1 calc_expectation(const real1 *probabilities, int size) {
    real1 expectation = 0;
    for (int i = 0; i < size; i++) {
        expectation += (i == 0 ? 1.0f : -1.0f) * probabilities[i];
    }
    return expectation;
}

real1 run_circuit(
    int input_size,
    const real1 *input,
    const real1 *thetas,
    int n_qubits,
    int levels
) {
    QInterfacePtr q_reg = CreateQuantumInterface(QINTERFACE_CPU, n_qubits, 0);

    // Set initial state
    q_reg->SetPermutation(0x0);

    // Dense angle encoding
    int i = 0;
    int is_short = input_size != n_qubits * 2;
    for (; i < n_qubits - (is_short ? 1 : 0); i++) {
        q_reg->RY(input[i], i);
        q_reg->RZ(input[i + 1], i);
    }
    if (is_short) {
        q_reg->RY(input[2 * i], i);
        q_reg->RZ(0, i);
    }

    // Apply TTN
    const real1 *first_theta = thetas;
    for (int j = 0; j < levels - 1; j++) {
        int pow = intPow(2, j);
        for (int k = n_qubits - 1; k >= pow - 1; k -= pow) {
            q_reg->RX(*first_theta++, k);
        }
        for (int k = n_qubits - 1; k - pow >= pow - 1; k -= 2 * pow) {
            q_reg->CNOT(k - pow, k);
        }
    }
    // Skip if one qubit only
    if (n_qubits > 1) {
        int last = n_qubits - 1 - intPow(2, levels - 1);
        q_reg->RX(*first_theta++, n_qubits - 1);
        // If n qubits is power of 2
        if (n_qubits % intPow(2, levels - 1) == 0) {
            q_reg->RX(*first_theta++, last);
        }
        q_reg->CNOT(last, n_qubits - 1);
    } else {
        q_reg->RX(*first_theta++, 0);
    }


    // Calc probabilities
    real1 probabilities[2] {};
    bitCapInt mask = (1 << (n_qubits - 1));
    q_reg->ProbMaskAll(mask, probabilities);

    real1 expectation = calc_expectation(probabilities, 2);

    return expectation;
}

extern "C"
void run_inputs(
    float *expectations,
    const real1 *inputs,
    const real1 *thetas,
    int n_inputs,
    int input_size
) {
    int n_qubits = (input_size + (2 - 1)) / 2;
    int levels = std::ceil(std::log2(n_qubits));

    #pragma omp parallel for default(none) shared( \
        expectations, n_inputs, input_size, inputs, thetas, n_qubits, levels \
    ) num_threads(CORES)
    for (int i = 0; i < n_inputs; i++) {
        expectations[i] = run_circuit(
            input_size, &inputs[i * input_size], thetas, n_qubits, levels
        );
    }
}


extern "C"
void run_thetas(
    float *expectations,
    float *inputs,
    float *thetas,
    int n_inputs,
    int input_size,
    int n_thetas
) {
    int n_qubits = (input_size + (2 - 1)) / 2; // Add normalization bit
    int theta_size = std::max(2 * n_qubits - 2, 1);

    float *theta_pointer = thetas;
    float *expectation_pointer = expectations;
    for (int i = 0; i < n_thetas; i++) {
        run_inputs(expectation_pointer, inputs, theta_pointer, n_inputs, input_size);
        theta_pointer += theta_size;
        expectation_pointer += n_inputs;
    }
}

int main() {
    int input_size = 2;
    auto input = new float[128]{
        0.02002, -0.0059,
        -0.00574, 0.00023,
        0.00762, 0.01041,
        -0.01526, 0.00107,
        0.03487, -0.01039,
        0.01813, 0.00455,
        -0.02060, -0.01876,
        -0.00387, -0.01271,
        0.01837, -0.02423,
        0.01944, 0.00552,
        0.01574, -0.00472,
        -0.01702, -0.00544,
        0.01747, -0.00013,
        -0.02561, -0.00234,
        -0.00558, -0.01674,
        0.00596, 0.00053,
        0.02630, -0.01280,
        -0.01043, 0.03037,
        -0.00428, -0.05031,
        -0.01515, -0.02582,
        0.02217, -0.01322,
        -0.01026, 0.03514,
        0.00090, -0.03042,
        -0.00970, -0.01330,
        -0.02130, -0.01341,
        0.01466, -0.00752,
        0.04476, -0.01595,
        -0.02057, -0.03465,
        -0.02475, 0.00358,
        0.01250, 0.00121,
        0.01807, 0.00946,
        0.04812, -0.01528,
        0.01192, -0.00290,
        0.00922, -0.00928,
        -0.00751, 0.00958,
        -0.02674, 0.00983,
        -0.01042, -0.00350,
        -0.00445, -0.00299,
        -0.02295, -0.01384,
        -0.02635, -0.04057,
        0.02803, -0.03035,
        0.01203, -0.01318,
        -0.03883, -0.03985,
        0.02933, -0.03130,
        -0.00580, -0.02749,
        -0.01576, -0.03959,
        0.01632, -0.01450,
        0.01566, -0.01623,
        0.01370, 0.01585,
        -0.01356, -0.04195,
        -0.04559, -0.03176,
        0.03980, 0.03008,
        0.03036, -0.02059,
        -0.00652, -0.02520,
        -0.01059, -0.02200,
        -0.00176, -0.03862,
        0.01058, 0.00739,
        0.02900, 0.02007,
        -0.00324, -0.00102,
        0.02504, -0.00565,
        0.00078, -0.02618,
        0.01552, -0.01023,
        -0.00826, -0.01351,
        0.01114, -0.00649
    };
    auto thetas = new float[1] { 0.38015 };
    float expectations[64];

    // Pre-calc eigenvalues
    int n_qubits = (input_size + (2 - 1)) / 2;
    int levels = std::ceil(std::log2(n_qubits));

    run_inputs(expectations, input, thetas, 64, 2);

    delete[] input;
    delete[] thetas;
}
