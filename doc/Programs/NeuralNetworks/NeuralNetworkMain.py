import sys, os

import time
import logging
import pickle

# Frameworks:
import numpy as np

#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=2'

import jax
import jaxlib
from jax.config import config
config.update("jax_enable_x64", True)
#config.update('jax_disable_jit', True)

# To specify the number of CPUs
#export XLA_FLAGS="--xla_force_host_platform_device_count=2"

import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev
from functools import partial

# Add the local folder to the import path:
top_folder = os.path.dirname(os.path.abspath(__file__))
top_folder = os.path.dirname(top_folder)
sys.path.insert(0,top_folder)

from NeuralWavefunction import Wavefunction
from Spin import Spin
from Metropolis import Metropolis
from NuclearPotential import NuclearPotential
from Observables import Observables
from Optimizer import Optimizer

jax.config.update('jax_threefry_partitionable', True)
#jax.config.update('jax_platform_name', 'cpu')#
#cpus = jax.devices("cpu")
#gpus = jax.devices("gpu")

#print ("cpus", cpus )
#print ("gpus", gpus )
#exit()

sigma0 = 2.0
dt = 0.002
neq = 10
nav = 10
nac = 1
nvoid = 100
nwalk = 3200
nopt = 1000
ndim = 3
nuc_name = '9Li'
seed_net = 19
seed_walk = 21
mass = 938.95
hbar = 197.327
delta = 0.0004
eps = 0.001
conf = 0.05
mix = 0.001
omega = 0.0
remove_cm = True
rec_r = True
rec_s = True
pot_name = 'pionless_lo_o'
pot_3b_name = 'R3_1.1'
module_load = False
module_write = True 
solver = 'CG'
n_devices = jax.local_device_count() 
print('n_devices=', n_devices)

print('jax device', jax.devices())
print('jax.device_count=', jax.device_count())
print('jax.local_device_count=', jax.local_device_count())

#exit()
# Model save
model_save_path = f"./{pot_name}_{nuc_name}.model"

# Set up logging:
logger = logging.getLogger()
# Create a file handler:
hdlr = logging.FileHandler(f'{pot_name}_{nuc_name}.log')
# Add formatting to the log:
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
ch = logging.StreamHandler()
logger.addHandler(hdlr) 
logger.addHandler(ch)
# Set the default level. Levels here: https://docs.python.org/2/library/logging.html
logger.setLevel(logging.DEBUG)

logger.info(f"jax version: {jax.__version__}")
logger.info(f"jaxlib version: {jaxlib.__version__}")
logger.info(f"nuc name: {nuc_name}")
logger.info(f"sigma0 = {sigma0}")
logger.info(f"dt = {dt}")
logger.info(f"neq = {neq}")
logger.info(f"nav = {nav}")
logger.info(f"nac = {nac}")
logger.info(f"nvoid = {nvoid}")
logger.info(f"nwalk = {nwalk}")
logger.info(f"nopt = {nopt}")
logger.info(f"ndim = {ndim}")
logger.info(f"seed_net = {seed_net}")
logger.info(f"seed_walk = {seed_walk}")
logger.info(f"mass = {mass}")
logger.info(f"hbar = {hbar}")
logger.info(f"delta = {delta}")
logger.info(f"eps = {eps}")
logger.info(f"conf = {conf}")
logger.info(f"mix = {mix}")
logger.info(f"remove center of mass = {remove_cm}")
logger.info(f"recover coordinates = {rec_r}")
logger.info(f"recover spin = {rec_s}")
logger.info(f"potential model = {pot_name}")
logger.info(f"3b potential model = {pot_3b_name}")
logger.info(f"solver = {solver}")

# Initialize Spin algebra
spin = Spin(nwalk, nuc_name)
npart = spin.npart 

# Initialize the network with one batch dimension, ndim, and npart
key = random.PRNGKey(seed_net)
key, key_input = jax.random.split(key)
wavefunction = Wavefunction(ndim, npart, conf, key_input, mix, spin, remove_cm)
params_init, nparams = wavefunction.build()
logger.info(f"number of parameters = {nparams}")

# Read saved params on file
if (module_load):
    with open(model_save_path, 'rb') as file:
         params_read = pickle.load(file)
# Mix with random parameters
    params = jax.tree_util.tree_map(wavefunction.update_mix, params_init, params_read)
else:
    params = params_init
#    params = jax.tree_util.tree_map(wavefunction.update_cut, params_read)
    
# Initialize Metropolis sampler
sigma = jnp.sqrt(hbar**2*dt/mass) 
metropolis = Metropolis(n_devices, nvoid, neq, nav, nac, nwalk, npart, ndim, sigma, sigma0, wavefunction, spin, remove_cm, logger)

# Initialize Potential
potential = NuclearPotential(nwalk, pot_name, pot_3b_name)

# Initialize Observables 
observables =  Observables(n_devices, mass, hbar, omega, ndim, npart, wavefunction, spin, potential, remove_cm)

# Initialize Optimizer
optimizer = Optimizer(n_devices, delta, eps, nparams, nwalk, npart, ndim, nav, nac, wavefunction, observables, spin, solver, remove_cm)

# Metropolis energy calculation
def vmc_run(key, neq, nav, nac, nvoid, params, rec_r, rec_s, it):

    tgen_i = time.time()
    key, key_o, r_o, sz_o = metropolis.initialize(key, rec_r, rec_s, it, params)
    Sz, Tz = observables.spin_isospin(sz_o)
    logger.info(f"initial S_z:, {Sz:3f}")
    logger.info(f"initial T_z:, {Tz:3f}")
    tgen_f = time.time()
    logger.info(f"Initial configurations generated, elapsed time: {tgen_f-tgen_i:.3f} seconds")

#    r_test = r_o.reshape((nwalk, npart, ndim))
#    sz_test = sz_o.reshape((nwalk, npart, 2))
#    energy = observables.energy(params, r_test, sz_test)
#    exit()

    twlk_i = time.time()
    r_stored, sz_stored, acc_r, acc_sz = metropolis.pmap_walk(params, r_o, sz_o, key_o)
    twlk_f = time.time()
    logger.info(f"Walk stored, elapsed time: {twlk_f-twlk_i:.3f} seconds")

    tav_i = time.time()
    obs_stored = jnp.zeros(shape=(n_devices, nwalk // n_devices, nav + nac, 2), dtype=jnp.complex128)
 
    for i in range (nav + nac):
        obs_pmap = observables.energy_pmap(params, r_stored[:,:,i,:,:], sz_stored[:,:,i,:,:])
        obs_stored = obs_stored.at[:,:,i,:].set(obs_pmap[:,:,:])

    obs_stored.block_until_ready()
    tav_f = time.time()
    logger.info(f"Observables computed, elapsed time: {tav_f-tav_i:.3f} seconds")

    acc_r = jnp.mean(acc_r)
    acc_sz = jnp.mean(acc_sz)
    obs_blk = jnp.mean(obs_stored, axis = (0, 1))
    obs_avg = jnp.mean(obs_blk, axis = 0)
    obs_err = jnp.std(obs_blk, axis = 0, ddof=1) / jnp.sqrt(nav+nac)

    energy = obs_avg[0]
    error = obs_err[0]
    energy_jf = obs_avg[1]
    error_jf = obs_err[1]

    delta_p = 0
    if (nopt > 1):
        delta_p = optimizer.sr(params, r_stored, sz_stored, obs_stored[:,:,:,0])
    else:
        delta_p = jax.tree_util.tree_map(wavefunction.update_zero, params)

    return energy, error, energy_jf, error_jf, acc_r, acc_sz, delta_p, key

# Optimization
key = random.PRNGKey(seed_walk)
for i in range(nopt):
    ti = time.time()
    energy, error, energy_jf, error_jf, acc_r, acc_sz, delta_p, key = vmc_run(key, neq, nav, nac, nvoid, params, rec_r, rec_s, i)
    tf = time.time()
      
    params = jax.tree_util.tree_map(wavefunction.update_add, params, delta_p)

    logger.info(f"step = {i}, energy = {energy:.3f}, err = {error:.3f}")
    logger.info(f"step = {i}, energy_jf = {energy_jf:.3f}, err = {error_jf:.3f}")
    logger.info(f"acceptance coordinates = {acc_r:.3f}")
    logger.info(f"acceptance spin = {acc_sz:.3f}")
    logger.info(f"elapsed time {tf - ti:.3f}")
    logger.info(f"\n")

# Write saved params on file
if (module_write):
   with open(model_save_path, 'wb') as file:
        pickle.dump(params, file)




