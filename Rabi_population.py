import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# System parameters
E1 = 1.0   # Energy for state |1⟩
E2 = 2.0   # Energy for state |2⟩
gamma = 1   # Amplitude of the time-dependent term --------- this value tunes the frequency of the two population states
omega_driving = 1  # Frequency of the time-dependent term ------ this value tunes the ampplitude difference between the two population states. They will be equal when omega_driving = E2-E1 (resonance)

# Operators
def H(t, args):
    H_0 = Qobj([[E1, 0], [0, E2]])
    H_1 = Qobj([[0, gamma * np.exp(1j * omega_driving * t)], [gamma * np.exp(-1j * omega_driving * t), 0]])  # Interaction Hamiltonian
    return Qobj(H_0 + H_1)

# Initial state: Ground state
psi0 = basis(2,0)

# Convert the pure state vector to a density matrix
rho0 = ket2dm(psi0)

# Time settings
t_list = np.linspace(0, 10, 100)

# Solve the von Neumann equation with the time-dependent Hamiltonian
result = mesolve(H, rho0, t_list)

# Extract the density matrix elements at each time step
density_matrix_elements = np.array([result.states[i].data.toarray() for i in range(len(result.states))])

# Extract the diagonal elements (population probabilities)
population_probabilities = np.real(np.diagonal(density_matrix_elements, axis1=-2, axis2=-1))

# Plot the population probabilities as a function of time
plt.figure(figsize=(10, 6))
plt.plot(t_list, population_probabilities[:, 0], label='|0⟩')
plt.plot(t_list, population_probabilities[:, 1], label='|1⟩')
plt.xlabel('Time')
plt.ylabel('Population Probability')
plt.legend()
plt.title('Population Probabilities as a function of time')
plt.show()