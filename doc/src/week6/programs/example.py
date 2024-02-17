import numpy as np

# Solution by brute force inversion
A = np.array([[2,1],[1,20]])
b = np.array([5,3])
invA = np.linalg.inv(A)
x = invA @ b
print(x)
print(np.linalg.eig(A))
#  Solution with iterative methods

# Solution with gradient descent and fixed learning rate

# Then using Jax
