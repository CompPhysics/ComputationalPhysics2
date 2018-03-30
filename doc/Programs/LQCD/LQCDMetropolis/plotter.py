import numpy as np, matplotlib.pyplot as plt

data = np.loadtxt("build/statistics.txt",skiprows=4)

file = open("build/statistics.txt")
lines = file.readlines()
acceptanceRate = lines[0].split()[-1]
NCor = lines[1].split()[-1]
NCf = lines[2].split()[-1]
NTherm = lines[3].split()[-1]

plt.errorbar(data[:,0],data[:,1],(data[:,2]),fmt="-o",ecolor="r",color="0")
plt.title("MC acceptance rate: %s, NCor=%s, NCf=%s, NTherm=%s" % (acceptanceRate,NCor,NCf,NTherm))
plt.xlabel(r"$t$",fontsize="18")
plt.ylabel(r"$\Delta E$",fontsize="18")
plt.grid()
plt.savefig("HO_metropolis0.png",dpi=300)
plt.show()

