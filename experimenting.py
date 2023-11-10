import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Parameters
omega_0 = 1.0  # qubit frequency

# Hamiltonian
H_0 = omega_0 * sigmaz() / 2.0

# Initial state
psi0 = basis(2, 0)

# Time evolution
times = np.linspace(0, 20, 1000)
result = mesolve(H_0, psi0, times, [], [sigmax(), sigmay(), sigmaz()])

# Plot coherences
plt.figure(figsize=(10, 6))
plt.plot(times, np.real(result.expect[1]), label=r'Re($\langle\sigma_y\rangle$)')
plt.plot(times, np.imag(result.expect[1]), label=r'Im($\langle\sigma_y\rangle$)')
plt.xlabel('Time')
plt.ylabel('Expectation Values')
plt.legend()
plt.show()
