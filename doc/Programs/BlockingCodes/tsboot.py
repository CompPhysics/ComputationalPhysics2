from numpy import *
from numpy.random import randint
from time import time


def tsboot(data,statistic,R,l):
    t = zeros(R); n = len(data); k = ceil(float(n)/l);
    inds = arange(n); t0 = time()
    
    # time series bootstrap
    for i in range(R):
        # construct bootstrap sample from
        # k chunks of data. The chunksize is l
        _data = concatenate([data[j:j+l] for j in randint(0,n-l,k)])[0:n];
        t[i] = statistic(_data)

    # analysis
    print "Runtime: %g sec" % (time()-t0); print "Bootstrap Statistics :"
    print "original           bias      std. error"
    print "%8g %14g %15g" % (statistic(data), \
                             mean(t) - statistic(data), \
                             std(t) )
    return t


# Demo

# data
X = loadtxt("resources/smalldata.txt")

# statistic to be estimated. Takes two args.
# arg1: the data
def stat(data):
    return mean(data)

t = tsboot(X, stat, 2**9, 2**10)
