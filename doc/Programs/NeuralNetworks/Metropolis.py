import time
import logging

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev
from jax.lax import fori_loop
from functools import partial


class Metropolis(object):
    """Metropolis Sampler in N dimension

    Sample from N-D coordinates, using the M(RT)^2 algorithm.
    It starts from a Gaussian distribution,  then propagates 
    for nvoid steps nobs times, and stores the walk

    """
    def __init__(self, n_devices, nvoid, neq, nav, nac, nwalk, npart, ndim, sigma, sigma0, wavefunction, spin, remove_cm, logger):
        self.n_devices = n_devices
        self.nvoid = nvoid
        self.neq = neq
        self.nav = nav
        self.nac = nac
        self.nsteps = (neq + nav + nac) * nvoid
        self.nwalk = nwalk
        self.npart = npart
        self.ndim = ndim
        self.wavefunction = wavefunction
        self.spin = spin
        self.npair = int(self.npart * (self.npart - 1) / 2)
        self.sigma = sigma 
        self.sigma0 = sigma0#1.2 * jnp.cbrt(self.npart / 3.)
        self.remove_cm = remove_cm
        self.logger = logger
        self.walk = jax.jit(self.walk)
        self.vmap_walk = jax.jit(self.vmap_walk)

    def initialize(self, key, recover_r, recover_s, it, params):
        if (recover_r and it > 0):
           r_o = self.r_s
        else:
           key, key_input = jax.random.split(key)
           r_o = self.sigma0 * jax.random.normal(key_input, shape=[self.nwalk, self.npart, self.ndim])
           r_o = self.split( r_o )
        if (self.remove_cm):
           rcm = jnp.mean(r_o, axis=2)
           r_o = r_o - rcm[:,:,None,:]
        if (recover_s and it > 0):
           sz_o = self.sz_s
        else:
           key, key_input = jax.random.split(key)
           r_o = r_o.reshape((self.nwalk, self.npart, self.ndim))
           sz_o = self.spin.sp_init(key_input, self.wavefunction, params, r_o)
           r_o = self.split( r_o )
           sz_o = self.split( sz_o )

        

        key, key_o = jax.random.split(key)
        key_o = jax.random.split(key_o, self.nwalk)
        key_o = self.split( key_o )

        return key, key_o, r_o, sz_o


# Performs the Metropolis walk using the fori loop constructions
    def walk(self, params, r_o, sz_o, key_o):

        # Single step fori_loop construct
        def step(i, loop_carry_i):
            r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s = loop_carry_i

            # Move coordinates with drift and reweight
            key_o, key_input = random.split(key_o)
            r_gauss = self.sigma * jax.random.normal(key_input, shape=[self.npart, self.ndim])
            r_n = r_o + r_gauss
            logpsi_n = self.wavefunction.logpsi(params, r_n, sz_o)
            prob = jnp.abs(jnp.exp(2 * (logpsi_n - logpsi_o)))
            key_o, key_input = random.split(key_o)
            unif = jax.random.uniform(key_input)
            accept = jnp.greater_equal(prob, unif)
            r_o = jnp.where(accept.reshape([1,1]), r_n, r_o)
            logpsi_o = jnp.where(accept, logpsi_n, logpsi_o)
            acc_s = acc_s.at[i // self.nvoid,0].set(accept.astype('float64'))

            # Swap spin and isospin of pairs ks and kt
            key_o, key_input = random.split(key_o)
            ks = jax.random.randint(key_input, shape = [1], minval = 0, maxval = self.npair)
            key_o, key_input = random.split(key_o)
            kt = jax.random.randint(key_input, shape = [1], minval = 0, maxval = self.npair)
            sz_n = self.spin.sp_prop(sz_o, ks, kt)
            logpsi_n = self.wavefunction.logpsi(params, r_o, sz_n)
            prob = jnp.abs(jnp.exp(2 * (logpsi_n - logpsi_o)))
            key_o, key_input = random.split(key_o)
            unif = jax.random.uniform(key_input)
            accept = jnp.greater_equal(prob, unif)
            sz_o = jnp.where(accept.reshape([1,1]), sz_n, sz_o)
            logpsi_o = jnp.where(accept, logpsi_n, logpsi_o)

            acc_s = acc_s.at[i // self.nvoid,1].set(accept.astype('float64'))
            r_s = r_s.at[i // self.nvoid,:,:].set(r_o)
            sz_s = sz_s.at[i // self.nvoid,:,:].set(sz_o)

            return r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s

        if (self.remove_cm):
           rcm = jnp.mean(r_o, axis=0)
           r_o = r_o - rcm[None,:]

        acc_s = jnp.zeros(shape=[self.neq + self.nav + self.nac, 2])
        r_s = jnp.zeros(shape=[self.neq + self.nav + self.nac, self.npart, self.ndim])
        sz_s = jnp.zeros(shape=[self.neq + self.nav + self.nac, self.npart, 2])

        logpsi_o = self.wavefunction.logpsi(params, r_o, sz_o)
        r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s = fori_loop(0, self.nsteps, step, (r_o, sz_o, key_o, logpsi_o, acc_s, r_s, sz_s) )

        acc_s = acc_s[self.neq:,:]
        r_s = r_s[self.neq:,:,:]
        sz_s = sz_s[self.neq:,:,:]

        return r_s, sz_s, acc_s

    def vmap_walk(self, params, r_o, sz_o, key_o):
        r_s, sz_s, acc_s = vmap(self.walk, in_axes=(None, 0, 0, 0)) (params, r_o, sz_o, key_o)
        return r_s, sz_s, acc_s

    def pmap_walk(self, params, r_pmap, sz_pmap, key_pmap):

        r_pmap, sz_pmap, acc_pmap = pmap(self.vmap_walk, in_axes=(None, 0, 0, 0))(params, r_pmap, sz_pmap, key_pmap)

        acc_pmap = acc_pmap.reshape((self.nwalk, self.nav+self.nac, 2))
        acc_r_pmap = jnp.mean(acc_pmap[:,:,0])
        acc_sz_pmap = jnp.mean(acc_pmap[:,:,1])

        self.r_s = r_pmap[:,:,self.nav + self.nac - 1,:,:]
        self.sz_s = sz_pmap[:,:,self.nav + self.nac - 1,:,:]

        return r_pmap, sz_pmap, acc_r_pmap, acc_sz_pmap

    def split(self, arr):
        return arr.reshape(self.n_devices, arr.shape[0] // self.n_devices, *arr.shape[1:]) 
