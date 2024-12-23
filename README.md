# QuantumForecaster
This simple quantum forecasting model uses quantum circuits to "learn" from time-series data. It's an introductory concept in quantum machine learning, and though it doesn’t outperform classical algorithms for such tasks, it's a fun way to explore quantum computing and demonstrate quantum-enhanced learning techniques.


Introduction

This project aims to use quantum circuits as a tool for forecasting time-series data. Specifically, we use Qiskit, a powerful open-source quantum computing framework, to build and simulate a quantum circuit. By optimizing the circuit parameters to minimize the Mean Squared Error (MSE) between predicted and true values, we explore how quantum circuits can be used in forecasting tasks.

The current implementation uses a simple sine wave as the time-series data for demonstration purposes, but it can be adapted to any time-series dataset.

Dependencies

To run this project, you'll need to install the following dependencies:

Qiskit: The core quantum computing library.
Qiskit Aer: The simulation framework for quantum circuits.
NumPy: For numerical operations.
SciPy: For optimization functions.
Matplotlib: For plotting results.