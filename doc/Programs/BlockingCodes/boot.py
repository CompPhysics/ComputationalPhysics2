from numpy import *
from numpy.random import randint
from time import time


def boot(data, statistic, R):
    t = zeros(R); n = len(data); inds = arange(n); t0 = time()
    
    # non-parametric bootstrap
    for i in range(R):
        t[i] = statistic(data[randint(0,n,n)])

    # analysis
    print "Runtime: %g sec" % (time()-t0); print "Bootstrap Statistics :"
    print "original           bias      std. error"
    print "%8g %14g %15g" % (statistic(data), \
                             mean(t) - statistic(data), \
                             std(t))
    return t

# Demo


X = loadtxt("resources/smalldata.txt")

# statistic to be estimated. Takes two args.
# arg1: the data

def stat(data):
    return mean(data)


# boot returns the bootstrap sample
t = boot(X, stat, 2**9)
