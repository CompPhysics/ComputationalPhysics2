import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("tau_int.dat")
a = data[:,0]
x = (100/a)**2

colors=["#d7191c", "#fdae61", "#abdda4", "#2b83ba"]
plt.errorbar(x, data[:,1], yerr=data[:,2],fmt=".", color=colors[0], capsize=2, elinewidth=1,markeredgewidth=0.5, label="Metropolis")
plt.errorbar(x+0.01*x, data[:,3], yerr=data[:,4],fmt=".", color=colors[1], capsize=2, elinewidth=1,markeredgewidth=0.5, label="HMC")
plt.errorbar(x+0.01*x, data[:,5]/100, yerr=data[:,6]/100,fmt=".", color=colors[3], capsize=2, elinewidth=1,markeredgewidth=0.5, label="Langevin/100")
plt.xlabel("$(L/a)^2$")
plt.ylabel(r"$\tau_{int}$")
plt.legend()
plt.savefig("tau_ints.pdf")
plt.show()

