import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

data = np.loadtxt('RMBenergy.dat')
x = data[:,0]
y = data[:,1]
plt.plot(x, y,'ro')
plt.axis([0,101,3, 5.5])
plt.xlabel(r'Iterations')
plt.ylabel(r'Energy')
plt.savefig('MLrbm.pdf')
plt.show()

