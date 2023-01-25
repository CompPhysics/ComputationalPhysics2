import numpy as np
import matplotlib.pyplot as plt
#from numba import jit
from tqdm import trange

# Constants
a = 0.2
m = 1
omega_sq = 1
T = 100
N = int(T/a)


samples = 100000
step_size = 0.025
corr = 20


# Lattice Action
#@jit(nopython=True)
def action(x):
    potential = 0.5 * omega_sq * x * x * a
    kinetic = np.zeros(len(x))
    for i in range(len(x)-1):
        kinetic[i] = m/(a*2) * ( (x[i+1] - x[i])**2)
    return np.sum(potential + kinetic)

def update(x):    
    x_new = x + np.random.uniform(-step_size,step_size, len(x))
    dS = action(x_new) - action(x)         
    if np.exp(-dS) > np.random.random():
        x = x_new            
        return x, 1
    return x, 0

def metropolis_sampling(x, N_conf, N_corr):
    acceptance = 0
    energy = np.zeros(N_conf)
    for i in trange(N_conf):
        for k in range(N_corr):
            x, accepted = update(x)
            acceptance += accepted
        energy[i] = np.average(x*x) 

    print(np.average(energy))
    print(np.std(energy))
    print(float(acceptance)/(N_corr*N_conf))

    plt.plot(range(N_conf), energy)
    plt.xlabel("Simuation Time")
    plt.xlabel("Energy")
    plt.title("Metropolis Sampling")
    plt.savefig("metropolis.pdf")
    plt.show()
    return energy

def tauInt(rho, N, lambdaMax=100):
    tauInt = 0.5
    for t in range(1, len(rho)):
        dRhoT = 0
        for k in range(t + lambdaMax):
            if k+t >= N/2:
                break
            dRhoT += (rho[t+k] + rho[np.abs(k-t)] - 2*rho[t]*rho[k])**2
        dRhoT /= N
        if np.sqrt(dRhoT) > rho[t]:
            tauInt += rho[t]
            break
        else:
            tauInt += rho[t]
    tauIntStd = np.sqrt(2*(2*t+1)/float(N))*tauInt
    return tauInt, tauIntStd

def autocorr(x):
    rho = np.zeros(int(len(x)/2))
    for k in range(int(len(x)/2)):
        rho[k] = np.corrcoef(np.array([x[0:len(x)-k], x[k:len(x)]]))[0,1]
    print(tauInt(rho, len(x)))

if __name__ == "__main__":
    initial_x = np.zeros(N)+np.sqrt(0.5)
    energy = metropolis_sampling(initial_x, samples, corr)
    autocorr(energy)
