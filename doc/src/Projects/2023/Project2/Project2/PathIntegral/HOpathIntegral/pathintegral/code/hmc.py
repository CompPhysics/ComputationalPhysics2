import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from tqdm import trange

# Constants
a = 0.2
m = 1
omega_sq = 1
T = 100
N = int(T/a)


samples = 100000
step_size = 0.0002
n_corr = 20
n_step_traj = 50

@jit(nopython=True)
def action(x):
    potential = 0.5 * omega_sq * x * x * a
    kinetic = np.zeros(len(x))
    for i in range(len(x)-1):
        kinetic[i] = m/(a*2) * ( (x[i+1] - x[i])**2)
    return np.sum(potential + kinetic)

@jit(nopython=True)
def action_derivative(x):
    potential_d = omega_sq * x * a
    kinetic_d = np.zeros(len(x))
    for i in range(len(x)):
        kinetic_d[i] = m / (a*2) * (-2*x[i] + x[i-1] + x[(i+1)%N]) 
    return potential_d + kinetic_d

@jit(nopython=True)
def leap_frog(x,p,n_step):
    p = p - step_size*0.5 * action_derivative(x)
    for i in range(1,n_step):
        x = x + step_size*p
        p = p - step_size * action_derivative(x)
    x = x + step_size*p
    p = p - step_size*0.5 * action_derivative(x)
    return x, p

def hmc(x, N_conf, N_corr):
    acceptance = 0
    energy = np.zeros(N_conf)
    for i in trange(N_conf):
        for k in range(N_corr):
            p = np.random.normal(0, 1, N)        
            x_new, p_new = leap_frog(x, p, n_step_traj)
            dK = np.sum(p_new*p_new)/2 - np.sum(p*p)/2
            dS = action(x_new) - action(x)            
            if np.exp(-(dS+dK)) > np.random.random():
                x = x_new
                acceptance+=1
        energy[i] = np.average(x*x) 


        #print(energy[sample])
        #print(x)
    print(np.average(energy))
    print(np.std(energy))
    print(float(acceptance)/(N_conf*N_corr))
    plt.plot(range(N_conf), energy)
    plt.xlabel("Simuation Time")
    plt.xlabel("Energy")
    plt.title("Hybrid Monte Carlo")
    plt.savefig("hmc.pdf")
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
    energy = hmc(initial_x, samples, n_corr)
    autocorr(energy)
