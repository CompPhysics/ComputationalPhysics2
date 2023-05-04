import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, jacfwd, jacrev, lax
from functools import partial

class Spin(object):
    def __init__(self, nwalk, nuc_name):
        self.nwalk = nwalk
        self.nuc_name = nuc_name
        self.nhid = 1
        if (self.nuc_name == '2H'):
           self.npart = 2
           self.nprot = 1
           self.nup = 2
        elif (self.nuc_name == '3He'):
           self.npart = 3
           self.nprot = 2
           self.nup = 2
        elif (self.nuc_name == '3H'):
           self.npart = 3
           self.nprot = 1
           self.nup = 2
        elif (self.nuc_name == '4He'):
           self.npart = 4
           self.nprot = 2
           self.nup = 2
        elif (self.nuc_name == '6Li'):
           self.npart = 6
           self.nprot = 3
           self.nup = 4
        elif (self.nuc_name == '6He'):
           self.npart = 6
           self.nprot = 2
           self.nup = 3
        elif (self.nuc_name == '7Li'):
           self.npart = 7
           self.nprot = 3
           self.nup = 4
        elif (self.nuc_name == '8Be'):
           self.npart = 8
           self.nprot = 4
           self.nup = 4
        elif (self.nuc_name == '9Li'):
           self.npart = 9
           self.nprot = 3
           self.nup = 4
        elif (self.nuc_name == '10Be'):
           self.npart = 10
           self.nprot = 4
           self.nup = 5
        elif (self.nuc_name == '12C'):
           self.npart = 12
           self.nprot = 6
           self.nup = 6
        elif (self.nuc_name == '16O'):
           self.npart = 16
           self.nprot = 8
           self.nup = 8
        elif (self.nuc_name == '40Ca'):
           self.npart = 40
           self.nprot = 20
           self.nup = 20
        elif (self.nuc_name == '2N'):
           self.npart = 2
           self.nprot = 0
           self.nup = 1
        else:
           print("Error, nuc_name name not valid", self.nuc_name)
        self.ndown = self.npart - self.nup
        self.npair = int(self.npart * (self.npart - 1) / 2)
        self.ip = jnp.empty(self.npair, dtype=int)
        self.jp = jnp.empty(self.npair, dtype=int)
        k = 0
        for i in range(self.npart-1):
            for j in range (i+1,self.npart):
                self.ip = self.ip.at[k].set(i)
                self.jp = self.jp.at[k].set(j)
                k+=1

    @partial(jit, static_argnums=(0,))
    def phi_spin(self, sz, Rnl_v, Rnl_h, rho_h):
        ph = jnp.zeros((self.npart + self.nhid, self.npart + self.nhid), dtype=jnp.complex128)
# Visible orbitals, visible coordinates
        for i in range(self.npart):
            ph = ph.at[i,0:self.npart].set( Rnl_v[i,:] )
        for i in range(self.nhid):
# Hidden orbitals, visible coordinates 
            ph = ph.at[self.npart+i,0:self.npart].set( Rnl_h[i,:] )
# Hidden orbitals, hidden coordinates 
            ph = ph.at[:,self.npart+i].set( rho_h[i,:] )
        sign, logdet = jnp.linalg.slogdet(ph)
        return sign, logdet

# Initialize the spin walk making sure the wave function is different from zero
    @partial(jit, static_argnums=(0, 2))
    def sp_init(self, key, wavefunction, params, r):
        def cond_fun(loop_carry):
            key, sz_o = loop_carry
            wphi_o = wavefunction.vmap_psi(params, r, sz_o)
            return jnp.min(jnp.abs(wphi_o)) == 0

        def body_fun(loop_carry):
            key, sz_o = loop_carry
            key, key_input = jax.random.split(key)
            sz_p = random.shuffle(key_input, sz_o, 1)
            wphi_p = wavefunction.vmap_psi(params, r, sz_p)
            accept = (wphi_p != 0)
            sz_o = jnp.where(accept.reshape([self.nwalk,1,1]), sz_p, sz_o)
            return key, sz_o


        sz_o = -1 * jnp.ones(shape=(self.nwalk, self.npart, 2))  
        sz_o = sz_o.at[:, 0:self.nup, 0].set(1)
        sz_o = sz_o.at[:, 0:self.nprot, 1].set(1)
        key, key_input = random.split(key)
        sz_o = random.shuffle(key_input, sz_o, 1)
        key, sz_o = lax.while_loop(cond_fun, body_fun, (key, sz_o))
        return sz_o

# Spin or isospin exchange operation
    @partial(jit, static_argnums=(0,))
    def sp_exch(self, sz_o, k, ist):
       sz_n = sz_o.at[self.ip[k],ist].set(sz_o[self.jp[k],ist])
       sz_n = sz_n.at[self.jp[k],ist].set(sz_o[self.ip[k],ist])
       return sz_n

# Batched version of it
    @partial(jit, static_argnums=(0,))
    def vmap_sp_exch(self, sz_o, k, ist):
       return vmap(self.sp_exch, in_axes=(0, 0, None))(sz_o, k, ist)

# Spin propagation by flipping the spin and the isospin of pairs ks and kt
    @partial(jit, static_argnums=(0,))
    def vmap_sp_prop(self, sz_o, ks, kt):
       sz_n = self.vmap_sp_exch(sz_o, ks, 0)
       sz_n = self.vmap_sp_exch(sz_n, kt, 1)
       return sz_n

    @partial(jit, static_argnums=(0,))
    def sp_prop(self, sz_o, ks, kt):
       sz_n = self.sp_exch(sz_o, ks, 0)
       sz_n = self.sp_exch(sz_n, kt, 1)
       return sz_n

