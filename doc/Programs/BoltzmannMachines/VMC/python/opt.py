# Gradient descent stepping with analytical derivative
import numpy as np
from scipy.optimize import minimize
def DerivativeE(x):
    return x-1.0/(4*x*x*x);

def Energy(x):
   return x*x*0.5+1.0/(8*x*x);
x0 = 1.0
eta = 0.1
Niterations = 100

for iter in range(Niterations):
    gradients = DerivativeE(x0)
    x0 -= eta*gradients

print(x0)
