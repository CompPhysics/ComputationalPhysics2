import sys, os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacfwd, jacrev
from functools import partial

class NuclearPotential(object):
    def __init__(self, nwalk, pot_name, pot_3b_name):
        self.nwalk=nwalk
        self.hc = 197.327053
        self.alpha = 1 / 137.03599
        self.b = 4.27
        if (pot_name == 'pionless_2'):
            self.vkr = 2.0
            self.v0r = -133.3431
            self.v0s = -9.0212
            self.ar3b = np.sqrt(68.48830)
            self.v_2b = self.pionless_2b
            self.v_3b = self.pionless_3b
        elif (pot_name == 'pionless_4'):
            self.vkr = 4.0
            self.v0r = -487.6128
            self.v0s = -17.5515
            self.ar3b = np.sqrt(677.79890)
            self.v_2b = self.pionless_2b
            self.v_3b = self.pionless_3b
        elif (pot_name == 'pionless_6'):
            self.vkr = 6.0
            self.v0r = -1064.5010
            self.v0s = -26.0830
            self.ar3b = np.sqrt(2652.65100)
            self.v_2b = self.pionless_2b
            self.v_3b = self.pionless_3b
        elif (pot_name == 'pionless_lo_a'):
            self.R0 = 1.7
            self.R1 = 1.5
            self.C01 = -4.38524414
            self.C10 = -8.00783936
            lamchi = 1000./self.hc
            fpi = 92.40/self.hc
            if (pot_3b_name == 'R3_1.0'):
               self.R3 = 1.0
               self.ce3b = 1.8354  
            elif (pot_3b_name == 'R3_1.5'):
               self.R3 = 1.5
               self.ce3b = 4.6301  
            elif (pot_3b_name == 'R3_2.0'):
               self.R3 = 2.0
               self.ce3b =  11.6871  
            elif (pot_3b_name == 'R3_2.5'):
               self.R3 = 2.5
               self.ce3b =  27.4702   
            else: 
               self.R3 = 1.
               self.ce3b =  0 
            self.ce3b = jnp.sqrt( self.ce3b / lamchi / fpi**4 * self.hc / jnp.pi**3 / self.R3**6 )
            self.v_2b = self.pionless_2b_lo
            self.v_3b = self.pionless_3b_lo
        elif (pot_name == 'pionless_lo_o'):
            self.R0 = 1.54592984
            self.R1 = 1.83039397
            self.C01 = -5.27518671
            self.C10 = -7.04040080
            lamchi = 1000./self.hc
            fpi = 92.40/self.hc
            if (pot_3b_name == 'R3_1.0'):
               self.R3 = 1.0
               self.ce3b = 1.0786 
            if (pot_3b_name == 'R3_1.1'):
               self.R3 = 1.1
               self.ce3b = 1.2945 
            elif (pot_3b_name == 'R3_1.5'):
               self.R3 = 1.5
               self.ce3b = 2.7676  
            elif (pot_3b_name == 'R3_2.0'):
               self.R3 = 2.0
               self.ce3b =  6.95356  
            elif (pot_3b_name == 'R3_2.5'):
               self.R3 = 2.5
               self.ce3b =  16.21993   
            else: 
               self.R3 = 1.
               self.ce3b =  0   
            self.ce3b = jnp.sqrt( self.ce3b / lamchi / fpi**4 * self.hc / jnp.pi**3 / self.R3**6 )   
            self.v_2b = self.pionless_2b_lo
            self.v_3b = self.pionless_3b_lo

        nr_test = 100
        r_test = jnp.linspace(0,5,nr_test)
        file_pot = open("pot_nn.dat", "w") 
        pot_test = np.zeros(shape=(8,nr_test))
        pot_test[0,:] = r_test
        for i in range(r_test.shape[0]):
            pot_test[1:7,i] =  self.v_2b(r_test[i])
            pot_test[7,i] =  self.v_em(r_test[i])
        np.savetxt(file_pot, np.transpose(pot_test), delimiter=' ', newline=os.linesep)
        file_pot.close()

    @partial(jit, static_argnums=(0,))
    def pionless_2b(self, rr):
        pot_2b=jnp.zeros(6)
        x = self.vkr * rr
        vr = jnp.exp( -x**2 / 4.0 )
        pot_2b = pot_2b.at[0].set(self.v0r * vr)
        pot_2b = pot_2b.at[2].set(self.v0s * vr)
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_3b(self, rr):
        x = self.vkr * rr
        vr = jnp.exp( -x**2 / 4.0 )
        pot_3b = self.ar3b * vr
        return pot_3b

    @partial(jit, static_argnums=(0,))
    def pionless_2b_lo(self, rr):
        pot_2b = jnp.zeros(6)
        C0_r = 1. / (jnp.sqrt(jnp.pi)*self.R0)**3*jnp.exp( -( rr / self.R0 )**2 )
        C1_r = 1. / (jnp.sqrt(jnp.pi)*self.R1)**3*jnp.exp( -( rr / self.R1 )**2 )
        pot_2b = pot_2b.at[0].set( 3. * ( self.C01 * C1_r + self.C10 * C0_r ) )
        pot_2b = pot_2b.at[1].set( self.C01 * C1_r - 3. * self.C10 * C0_r )
        pot_2b = pot_2b.at[2].set( -3. * self.C01 * C1_r + self.C10 * C0_r )
        pot_2b = pot_2b.at[3].set( -1. * ( self.C01 * C1_r + self.C10 * C0_r ) )
        pot_2b = pot_2b / 16. * self.hc
        return pot_2b

    @partial(jit, static_argnums=(0,))
    def pionless_3b_lo(self, rr):
        pot_3b = self.ce3b * jnp.exp( - ( rr / self.R3 )**2 )
        return pot_3b

    @partial(jit, static_argnums=(0,))
    def v_em(self, rr):
        rr = jnp.maximum(rr,0.0001)
        br = self.b * rr
        fcoul = 1 - (1 + 11 * br / 16 + 3 * br**2 / 16 + br**3 / 48) * jnp.exp(-br)
        pot_em = self.alpha * self.hc * fcoul / rr
        return pot_em


