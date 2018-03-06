from numpy import *
from numpy.random import randint
from time import time


def jack(data, stat):
    n = len(data);t = zeros(n); inds = arange(n); t0 = time()
    # non-parametric bootstrap
    for i in range(n):
        t[i] = stat(delete(data,i) )

    # analysis
    print "Runtime: %g sec" % (time()-t0); print "Jackknife Statistics :"
    print "original           bias      std. error"
    print "%8g %14g %15g" % (stat(data), \
                             (n-1)*(mean(t) - stat(data)), \
                             (n*var(t))**.5)
    return t

# Demo


X = loadtxt("resources/smalldata.txt")

def stat(data):
    return mean(data)

# boot returns the bootstrap sample
t = jack(X, stat)
