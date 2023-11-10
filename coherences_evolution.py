import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
E1 = 1
E2 = 2

# Operators
H_0 = Qobj([[E1, 0], [0, E2]])

# Initial pure state vector
psi0 = basis(2, 0) + basis(2,1)

# Convert the pure state vector to a density matrix
rho0 = ket2dm(psi0)

# Time evolution
times = np.linspace(0, 5, 1000)
result = mesolve(H_0, rho0, times)

# Extract the density matrix elements at each time step
density_matrix_elements = np.array([result.states[i].full() for i in range(len(result.states))])

# Extract the diagonal elements (population probabilities)
population_probabilities = np.real(np.diagonal(density_matrix_elements, axis1=-2, axis2=-1))

# Extract the off-diagonal elements (coherences)
coherences = np.array([density_matrix_elements[i, 0, 1] for i in range(len(result.states))])

# Plot the coherences as a function of time
plt.figure(figsize=(10, 6))
plt.plot(times, coherences.real, label='real-coherence')
plt.plot(times, coherences.imag, label='Imaginary coherence')
plt.xlabel('Time')
plt.ylabel('Coherence')
plt.legend()
plt.title('Coherences as a function of time')
plt.show()
