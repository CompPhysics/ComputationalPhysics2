#!/usr/bin/env python
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random

# initialize the rng with a seed, simple uniform distribution
random.seed() 
m = 1000
samplefactor = 1.0/m
x = np.zeros(m)   
MeanValue = 0.
VarValue = 0.
for i in range (m):
    value = random.random()
    x[i] = value
    MeanValue += value
    VarValue += value*value

MeanValue *= samplefactor
VarValue *= samplefactor
Variance = VarValue-MeanValue*MeanValue
STDev = np.sqrt(Variance)
print("MeanValue =", MeanValue)
print("Variance =", Variance)
print("Standard deviation =", STDev)

# Computing the autocorrelation function
autocorrelation = np.zeros(m)
darray = np.zeros(m)
for j in range (m):
    sum = 0.0
    darray[j] = j
    for k in range (m-j):
        sum += (x[k]-MeanValue)*(x[k+j]-MeanValue ) 
    autocorrelation[j] = (sum/Variance)*samplefactor
# print	variance with autocorrelation
print("Final variance = ", Variance*(1.0+np.sum(autocorrelation)))
# Visualize results
plt.plot(darray, autocorrelation,'ro')
plt.axis([0,m,-0.2, 1.1])
plt.xlabel(r'$d$')
plt.ylabel(r'$\kappa_d$')
plt.title(r'autocorrelation function for RNG with uniform distribution')
plt.show()

