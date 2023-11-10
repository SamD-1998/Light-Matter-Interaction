from cmath import pi
import matplotlib.pyplot as plt
import numpy as np
from qutip import *

# DEFINING THE PROBLEM PARAMETERS

wc = 1.0 * 2 * pi       # cavity frequency
wa = 1.0 * 2 * pi       # atom frequency
g = 0.05 * 2 * pi       # coupling strength
kappa = 0.05 * 2 * pi   # cavity dissipation rate
gamma = 0.05            # atom dissipation rate
N = 15                  # Number of cavity Fock states
n_th_a = 0.0            # avg number of thermal bath excitations
use_rwa = True

tlist = np.linspace(0,25,101)       # discrete time points at which information of the state of the system is derived

# SETUP THE OPERATORS, THE HAMILTONIAN AND THE INITIAL STATE

# initial state
psi0 = tensor(basis(N,0), basis(2,1))   # start with an excited state

# operators
a = tensor(destroy(N), qeye(2))
sm = tensor(qeye(N), destroy(2))

# Hamiltonian

if use_rwa:
    H = wc*a.dag()*a + wa*sm.dag()*sm + g*(a.dag()*sm + a*sm.dag())
else:
    H = wc*a.dag()*a + wa*sm.dag()*sm + g*(a + a.dag())*(sm + sm.dag())

# CREATE A LIST OF COLLAPSE OPERATORS THAT DESCRIBE DISSIPATION

c_ops = []          # List of collapse operators

# cavity relaxtion
rate = kappa * (1+ n_th_a)
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a)

# cavity excitation, if temperature > 0
rate = kappa * n_th_a
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * a.dag())

# qubit relaxation
rate = gamma
if rate > 0.0:
    c_ops.append(np.sqrt(rate) * sm)

# EVOLVE THE SYSTEM

output = mesolve(H, psi0, tlist, c_ops, [a.dag()*a, sm.dag()*sm])

# VISUALIZING THE RESULTS

n_c = output.expect[0]
n_a = output.expect[1]

fig, axes = plt.subplots(1,1,figsize = (10,6))

axes.plot(tlist, n_c, label = "Cavity")
axes.plot(tlist, n_a, label = "Atom excited state")
axes.legend(loc = 0)
axes.set_xlabel("Time")
axes.set_ylabel("Occupation Probability")
axes.set_title("Vacuum Rabi Oscillations")
plt.show()