# Common imports
import os

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "Results/VMCHydrogen"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')

outfile = open(data_path("VMCHydrogen.dat"),'w')


# VMC for the hydrogen atom
# Brute force Metropolis, no importance sampling and no energy minimization
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# Trial wave function for the hydrogen atom
def WaveFunction(r,alpha):
    argument = np.linalg.norm(r)
    return alpha*argument*exp(-alpha*argument)

# Local energy  for the hydrogen atom
def LocalEnergy(r,alpha):
    return (1.0/np.linalg.norm(r))*(alpha-1)-alpha*alpha*0.5

# The Monte Carlo sampling with the Metropolis algo
# The jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when the function is called.
@jit
def MonteCarloSampling():

    NumberMCcycles= 10000
    StepSize = 1.0
    # positions
    PositionOld = np.zeros((Dimension), np.double)
    PositionNew = np.zeros((Dimension), np.double)

    # seed for rng generator
    seed()
    # start variational parameter
    alpha = 0.85
    for ia in range(MaxVariations):
        alpha += .025
        AlphaValues[ia] = alpha
        energy = energy2 = 0.0
        #Initial position
        for j in range(Dimension):
            PositionOld[j] = StepSize * (random() - .5)
        wfold = WaveFunction(PositionOld,alpha)
        #Loop over MC MCcycles
        for MCcycle in range(NumberMCcycles):
            #Trial position 
            for j in range(Dimension):
                PositionNew[j] = PositionOld[j] + StepSize*(random() - .5)
            wfnew = WaveFunction(PositionNew,alpha)
            #Metropolis test to see whether we accept the move
            if random() <= wfnew**2 / wfold**2:
                PositionOld = PositionNew
                wfold = wfnew
            DeltaE = LocalEnergy(PositionOld,alpha)
            energy += DeltaE
            energy2 += DeltaE**2
        #We calculate mean, variance and error
        energy /= NumberMCcycles
        energy2 /= NumberMCcycles
        variance = energy2 - energy**2
        error = sqrt(variance/NumberMCcycles)
        Energies[ia] = energy    
        Variances[ia] = variance    
        outfile.write('%f %f %f %f \n' %(alpha,energy,variance,error))
    return Energies, AlphaValues, Variances


#Here starts the main program with variable declarations
NumberParticles = 1
Dimension = 3
MaxVariations = 10
Energies = np.zeros((MaxVariations))
Variances = np.zeros((MaxVariations))
AlphaValues = np.zeros(MaxVariations)
ExactEnergy = np.zeros((MaxVariations))
(Energies, AlphaValues, Variances) = MonteCarloSampling()
ExactEnergy = AlphaValues*(AlphaValues*0.5-1.)
outfile.close()
#simple subplot
plt.subplot(2, 1, 1)
plt.plot(AlphaValues, Energies, 'o-',AlphaValues, ExactEnergy, 'r-')
plt.title('Energy and variance')
plt.ylabel('Dimensionless energy')
plt.subplot(2, 1, 2)
plt.plot(AlphaValues, Variances, '.-')
plt.xlabel(r'$\alpha$', fontsize=15)
plt.ylabel('Variance')
save_fig("VMCHydrogen")
plt.show()
#nice printout with Pandas
import pandas as pd
from pandas import DataFrame
data ={'Alpha':AlphaValues, 'Energy':Energies,'ExactEnergy':ExactEnergy,'Variance':Variances}

frame = pd.DataFrame(data)
print(frame)





