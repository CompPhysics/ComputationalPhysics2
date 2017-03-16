import numpy as np
import matplotlib.pyplot as plt

def block_mean(vec):
    return sum(vec)/len(vec)

def meanAndVariance(vec):
    mean = sum(vec)/len(vec)
    var = sum([i ** 2 for i in vec])/len(vec) - mean*mean
    return mean, var

data = [float( line.rstrip('\n')) for line in open('energy.txt')]

n_blocks = 200
block_size_min = 100
block_size_max = len(data)/100
block_step = int ((block_size_max - block_size_min + 1) / n_blocks)

mean_vec = []
var_vec = []
block_sizes = []
for i in range(0, n_blocks):
    mean_temp_vec = []
    start_point = 0
    end_point = block_size_min + block_step*i
    block_size = end_point
    block_sizes.append(block_size)

mean_temp_vec.append(block_mean(data[start_point:end_point]))
start_point = end_point
end_point += block_size_min + block_step*i
mean, var = meanAndVariance(mean_temp_vec)
mean_vec.append(mean)
var_vec.append(np.sqrt(  var/(len(data)/float(block_size) - 1.0)     ))

mean, var = meanAndVariance(data)
