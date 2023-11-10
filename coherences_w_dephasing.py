import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
E1 = 1
E2 = 2
qubit_relaxation = 0.02  # qubit relaxation rate
dephasing_rate = 0.01  # dephasing rate

# Operators
H_0 = Qobj([[E1, 0], [0, E2]])

# Collapse operators
c_ops = [np.sqrt(qubit_relaxation) * sigmam(), np.sqrt(dephasing_rate) * sigmaz()]

# Initial state
psi0 = basis(2, 0) + basis(2,1)

# Convert the pure state vector to a density matrix
rho0 = ket2dm(psi0)

# Time evolution
times = np.linspace(0, 20, 1000)
result = mesolve(H_0, rho0, times, c_ops)

# Extract the density matrix elements at each time step
density_matrix_elements = np.array([result.states[i].full() for i in range(len(result.states))])

# Extract the diagonal elements (population probabilities)
population_probabilities = np.real(np.diagonal(density_matrix_elements, axis1=-2, axis2=-1))

# Extract the off-diagonal elements (coherences)
coherences = np.array([density_matrix_elements[i, 0, 1] for i in range(len(result.states))])

# Plot coherences
plt.figure(figsize=(10, 6))
plt.plot(times, np.real(result.expect[1]), label=r'Re($\langle\sigma_y\rangle$)')
plt.plot(times, np.imag(result.expect[1]), label=r'Im($\langle\sigma_y\rangle$)')
plt.xlabel('Time')
plt.ylabel('Expectation Values')
plt.legend()
plt.show()
