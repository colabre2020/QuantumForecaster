import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator  # Correct import for AerSimulator
from scipy.optimize import minimize
from qiskit.circuit import Parameter

# Step 1: Generate a Simple Time-Series Dataset (Sine Wave)
def generate_sine_wave(num_points=100):
    x = np.linspace(0, 4 * np.pi, num_points)  # Time points
    y = np.sin(x)  # Sine wave as the target time series
    return x, y

# Generate the dataset
x_data, y_data = generate_sine_wave(100)

# Plot the dataset
plt.plot(x_data, y_data, label="Sine Wave (Time Series)")
plt.title("Generated Sine Wave")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# Step 2: Define a Quantum Circuit for Regression
def create_quantum_circuit():
    # Define the number of qubits (3 qubits)
    qc = QuantumCircuit(3)
    
    # Define parameters (angles) that will be optimized
    theta1 = Parameter('θ1')
    theta2 = Parameter('θ2')
    theta3 = Parameter('θ3')
    
    # Apply gates to the qubits (create a simple parameterized quantum circuit)
    qc.h(0)  # Apply Hadamard gate to the first qubit
    qc.rx(theta1, 1)  # Apply RX rotation to the second qubit
    qc.rx(theta2, 2)  # Apply RX rotation to the third qubit
    qc.cx(0, 1)  # Apply CNOT gate (control qubit 0, target qubit 1)
    qc.cx(1, 2)  # Apply CNOT gate (control qubit 1, target qubit 2)
    
    # Measure the qubits
    qc.measure_all()
    
    return qc, [theta1, theta2, theta3]

# Step 3: Define a Cost Function for the Quantum Circuit (MSE)
def cost_function(params, quantum_circuit, simulator, x_data, y_data):
    # Bind the parameters to the quantum circuit
    parameter_dict = {quantum_circuit.parameters[i]: params[i] for i in range(len(params))}
    bound_circuit = quantum_circuit.bind_parameters(parameter_dict)
    
    # Simulate the circuit
    job = simulator.run(bound_circuit, shots=1)
    result = job.result()
    
    # Get the outcome (measurement result)
    counts = result.get_counts()
    outcome = list(counts.keys())[0]
    
    # Convert the outcome to a value (scaled down for 3 qubits)
    predicted_value = int(outcome, 2) / 8
    
    # Compute the MSE between the predicted value and true value
    mse = np.mean((predicted_value - y_data) ** 2)
    return mse

# Step 4: Optimize the Quantum Circuit Parameters Using Scipy
def train_quantum_circuit(x_data, y_data, quantum_circuit):
    # Create the simulator
    simulator = AerSimulator()  # Use AerSimulator from qiskit_aer
    
    # Initial random parameters for the quantum circuit
    initial_params = np.random.rand(len(quantum_circuit[1]))  # Random initial parameters
    
    # Minimize the cost function (MSE)
    result = minimize(cost_function, initial_params, args=(quantum_circuit[0], simulator, x_data, y_data), method='COBYLA')
    
    return result

# Step 5: Forecast Future Values Using the Optimized Parameters
def forecast_future_values(quantum_circuit, optimized_params, future_points=10):
    forecast = []
    
    # Create the simulator
    simulator = AerSimulator()  # Use AerSimulator from qiskit_aer
    
    for _ in range(future_points):
        # Bind the optimized parameters to the circuit
        parameter_dict = {quantum_circuit[0].parameters[i]: optimized_params[i] for i in range(len(optimized_params))}
        circuit = quantum_circuit[0].bind_parameters(parameter_dict)
        
        # Simulate the circuit and get the predicted value
        job = simulator.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        outcome = list(counts.keys())[0]
        
        # Convert the outcome to a value
        predicted_value = int(outcome, 2) / 8  # Scale the value to match the range
        
        forecast.append(predicted_value)
    
    return forecast

# Step 6: Train the Quantum Circuit and Forecast Future Values
quantum_circuit, params = create_quantum_circuit()
result = train_quantum_circuit(x_data, y_data, quantum_circuit)

# Print the optimized parameters
print("Optimized Parameters:", result.x)

# Step 7: Forecast the Next 10 Points (Future Time-Series Values)
forecasted_values = forecast_future_values(quantum_circuit, result.x, future_points=10)

# Plot the original time series and forecasted values
plt.plot(x_data, y_data, label="Original Time Series")
plt.plot(np.arange(len(x_data), len(x_data) + 10), forecasted_values, label="Forecasted Values", linestyle='--')
plt.legend()
plt.title("Quantum Time-Series Forecasting")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
