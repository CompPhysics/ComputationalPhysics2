import sys, os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacfwd, jacrev
from functools import partial

class CoulombPotential(object):
    def __init__(self, nwalk, pot_name):
        self.nwalk=nwalk
        self.hc = 197.327053
        self.alpha = 1 / 137.03599
        self.b = 4.27

        nr_test = 100
        r_test = jnp.linspace(0,5,nr_test)
        file_pot = open("pot_coul.dat", "w") 
        pot_test = np.zeros(shape=(8,nr_test))
        pot_test[0,:] = r_test
        for i in range(r_test.shape[0]):
            pot_test[7,i] =  self.v_em(r_test[i])
        np.savetxt(file_pot, np.transpose(pot_test), delimiter=' ', newline=os.linesep)
        file_pot.close()


    @partial(jit, static_argnums=(0,))
    def v_coul(self, rr):
        rr = jnp.maximum(rr,0.0001)
        br = self.b * rr
        # rewrite this
        fcoul = 1 - (1 + 11 * br / 16 + 3 * br**2 / 16 + br**3 / 48) * jnp.exp(-br)
        pot_coul = self.alpha * self.hc * fcoul / rr
        return pot_coul


