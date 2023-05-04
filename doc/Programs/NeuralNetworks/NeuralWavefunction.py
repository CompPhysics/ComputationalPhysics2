import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev
from jax.tree_util import tree_flatten
from jax.flatten_util import ravel_pytree
from jax.example_libraries import stax
from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, Gelu, LogSoftmax, Softplus, Tanh, 
                                   Sigmoid, elementwise, FanOut, FanInConcat)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

from functools import partial
import pickle

@jit
def leakytanh(x):
    act = 0.975
    return act * jnp.tanh(x) + (1 - act) * x 

LeakyTanh = elementwise(leakytanh)
Sin = elementwise(jnp.sin)

class Wavefunction(object):
    def __init__(self, ndim, npart, conf, key, mix, spin, remove_cm):
        self.ndim = ndim + 2
        self.npart = npart
        self.conf = conf
        self.key = key
        self.mix = mix
        self.spin = spin
        self.remove_cm = remove_cm
        self.ndense = 32
        self.nsingle = 16
        self.nlat = 32
        self.activation = LeakyTanh
        self.a = 8
        self.nhid = 1
        self.spin.nhid = self.nhid
        self.R0 = 1.2 * jnp.cbrt(self.npart) / jnp.sqrt( 3 )

        self.npair = self.npart * (self.npart - 1)
        self.k = jnp.arange(self.npair)
        self.ip = jnp.empty(self.npair, dtype=int)
        self.jp = jnp.empty(self.npair, dtype=int)
        k = 0
        for i in range(self.npart):
            for j in range (self.npart):
                if (i != j):
                   self.ip = self.ip.at[k].set(i)
                   self.jp = self.jp.at[k].set(j)
                   k += 1
        self.vmap_psi = jax.jit(self.vmap_psi)

    def build(self):
# Deep-Sets: m
        self.m_init, self.m_apply = stax.serial(
        Dense(self.ndense), self.activation,
        Dense(self.ndense), self.activation,
        Dense(self.nlat - self.ndim),
)

# Deep-Sets: phi
        self.phi_init, self.phi_apply = stax.serial(
        Dense(self.ndense), self.activation,
        Dense(self.ndense), self.activation,
        Dense(self.nlat),
)

# Deep-Sets: rho visible
        rho_v_ff = stax.serial(
        Dense(self.ndense), self.activation,
        Dense(self.ndense), self.activation,
        Dense(self.npart),
)

# Deep-Sets: rho hidden
        rho_h_ff = stax.serial(
        Dense(self.ndense), self.activation,
        Dense(self.ndense), self.activation,
        Dense(self.nhid),
)

# Deep-Sets: combined rho hidden
        self.rho_init, self.rho_apply = stax.serial(
        FanOut(2),
        stax.parallel(rho_v_ff, rho_h_ff),
        FanInConcat()
)

# Feed-forward constructor for single-particle orbitals wave function  
        self.orb_init, self.orb_apply = stax.serial(
        Dense(self.nsingle), self.activation,
        Dense(self.nsingle), self.activation,
        Dense(1),
)

# Create self.nhid parallel structures of rho(phi(x))
        x_in_shape = ( -1, self.ndim )
        x_ij_in_shape = ( -1, 2 * self.ndim ) 
        x_i_in_shape =  ( -1, self.nlat )
        rho_in_shape = ( -1, self.nlat ) 

# Initialize message network
        self.key, key_input = jax.random.split(self.key)
        _, m_params = self.m_init(key_input, x_ij_in_shape) 
        self.num_m_params = len(m_params)

        phi_p_params=[]
        phi_a_params=[]
        rho_p_params=[]
        rho_a_params=[]
        orb_h_p_params=[]
        orb_h_a_params=[]
        for i in range(self.nhid):
            self.key, key_input = jax.random.split(self.key)
            _, phi_p_params_i = self.phi_init(key_input, x_i_in_shape) 
            phi_p_params.append(phi_p_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, phi_a_params_i = self.phi_init(key_input, x_i_in_shape) 
            phi_a_params.append(phi_a_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, rho_p_params_i = self.rho_init(key_input, rho_in_shape) 
            rho_p_params.append(rho_p_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, rho_a_params_i = self.rho_init(key_input, rho_in_shape) 
            rho_a_params.append(rho_a_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, orb_h_p_params_i = self.orb_init(key_input, x_i_in_shape) 
            orb_h_p_params.append(orb_h_p_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, orb_h_a_params_i = self.orb_init(key_input, x_i_in_shape) 
            orb_h_a_params.append(orb_h_a_params_i)

        orb_v_p_params=[]
        orb_v_a_params=[]
        for i in range(self.npart):
            self.key, key_input = jax.random.split(self.key)
            _, orb_v_p_params_i = self.orb_init(key_input, x_i_in_shape) 
            orb_v_p_params.append(orb_v_p_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, orb_v_a_params_i = self.orb_init(key_input, x_i_in_shape) 
            orb_v_a_params.append(orb_v_a_params_i)

        self.num_phi_p_params = len(phi_p_params)
        self.num_phi_a_params = len(phi_a_params)
        self.num_rho_p_params = len(rho_p_params)
        self.num_rho_a_params = len(rho_a_params)
        self.num_orb_h_p_params = len(orb_h_p_params)
        self.num_orb_h_a_params = len(orb_h_a_params)
        self.num_orb_v_p_params = len(orb_v_p_params)
        self.num_orb_v_a_params = len(orb_v_a_params)

        net_params = [m_params] + [phi_p_params] + [phi_a_params] + [rho_p_params] + [rho_a_params] + [orb_h_p_params] + [orb_h_a_params] + [orb_v_p_params] + [orb_v_a_params]

# cast to double
        net_params = jax.tree_util.tree_map(self.update_cast, net_params)
        flat_net_params = self.flatten_params(net_params)
        num_flat_params = flat_net_params.shape[0]

        with open('full.model', 'wb') as file:
             pickle.dump(net_params, file)

        return net_params, num_flat_params

# graph net for single-particle coordinates
    @partial(jit, static_argnums=(0,))
    def x_pair(self, k, x):
        x_ij = jnp.concatenate((x[self.ip[k],:],x[self.jp[k],:]))
        return x_ij

    @partial(jit, static_argnums=(0,))
    def logsignpsi(self, params, r, sz):

        m_params, phi_p_params, phi_a_params, rho_p_params, rho_a_params, orb_h_p_params, orb_h_a_params, orb_v_p_params, orb_v_a_params = params

#        num_offset_params = 0

#        m_params = params[num_offset_params : num_offset_params + self.num_m_params]
#        num_offset_params = num_offset_params + self.num_m_params

#        phi_p_params = params[num_offset_params : num_offset_params + self.num_phi_p_params]
#        num_offset_params = num_offset_params + self.num_phi_p_params

#        phi_a_params = params[num_offset_params : num_offset_params + self.num_phi_a_params]
#        num_offset_params = num_offset_params + self.num_phi_a_params

#        rho_p_params = params[num_offset_params : num_offset_params + self.num_rho_p_params]
#        num_offset_params = num_offset_params + self.num_rho_p_params

#        rho_a_params = params[num_offset_params : num_offset_params + self.num_rho_a_params]
#        num_offset_params = num_offset_params + self.num_rho_a_params

#        orb_h_p_params = params[num_offset_params : num_offset_params + self.num_orb_h_p_params]
#        num_offset_params = num_offset_params + self.num_orb_h_p_params

#        orb_h_a_params = params[num_offset_params : num_offset_params + self.num_orb_h_a_params]
#        num_offset_params = num_offset_params + self.num_orb_h_a_params

#        orb_v_p_params = params[num_offset_params : num_offset_params + self.num_orb_v_p_params]
#        num_offset_params = num_offset_params + self.num_orb_v_p_params

#        orb_v_a_params = params[num_offset_params : num_offset_params + self.num_orb_v_a_params]
#        num_offset_params = num_offset_params + self.num_orb_v_a_params

        if (self.remove_cm):
           rcm = jnp.mean(r, axis=0)
           r = ( r - rcm[None,:] ) 
        r = r / self.R0
        x = jnp.concatenate((r, sz ), axis=1)

        x_ij = vmap(self.x_pair, in_axes=(0, None)) (self.k, x)
        m_ij = jnp.reshape(self.m_apply(m_params, x_ij), newshape=[self.npart, self.npart-1, -1])
        x_i = jnp.concatenate((x, jnp.mean(m_ij, axis=1) ), axis=1) 
        
# rho_h is (nhid, npart + nhid ) where I concatenate rho(npart) and rho(nhid)
        phi_p_params_stacked = self.pytrees_stack(phi_p_params)
        phi_a_params_stacked = self.pytrees_stack(phi_a_params)
        rho_p_params_stacked = self.pytrees_stack(rho_p_params)
        rho_a_params_stacked = self.pytrees_stack(rho_a_params)
        rho_h = vmap(self.phi_chi_h, in_axes=(0, 0, 0, 0, None))(phi_p_params_stacked, phi_a_params_stacked, rho_p_params_stacked, rho_a_params_stacked, x_i)

# orb_v_p is (nstate=npart, npart)
        orb_v_p_params_stacked = self.pytrees_stack(orb_v_p_params)
        orb_v_p = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_v_p_params_stacked, x_i), newshape=[self.npart, self.npart])

        orb_v_a_params_stacked = self.pytrees_stack(orb_v_a_params)
        orb_v_a = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_v_a_params_stacked, x_i), newshape=[self.npart, self.npart])

        orb_v = jnp.exp( self.a * jnp.tanh( orb_v_a / self.a ) + 1j * orb_v_p )

# orb_h_p is (nstate=nhid, npart)
        orb_h_p_params_stacked = self.pytrees_stack(orb_h_p_params)
        orb_h_p = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_h_p_params_stacked, x_i), newshape=[self.nhid, self.npart])

        orb_h_a_params_stacked = self.pytrees_stack(orb_h_a_params)
        orb_h_a = jnp.reshape(vmap(self.orb_apply, in_axes=(0, None))(orb_h_a_params_stacked, x_i), newshape=[self.nhid, self.npart])

        orb_h = jnp.exp( self.a * jnp.tanh( orb_h_a / self.a ) + 1j * orb_h_p )

        sign, logdet = self.spin.phi_spin(sz, orb_v, orb_h, rho_h)
        log = logdet -  self.conf * jnp.sum( r**2 )

        sign = jnp.imag(jnp.log(sign))
        sign = jnp.reshape(sign, ())
        log = jnp.reshape(log, ())
        return sign, log  

    @partial(jit, static_argnums=(0,))
    def logpsi(self, params, r, sz):
        sz_m = sz.at[:,0].set(-sz[:,0])
#        sz_pm = jnp.stack((sz, sz_m, sz, sz_m))
#        r_pm = jnp.stack((r, r, -r, -r))
        sz_pm = jnp.stack((sz, sz))
        r_pm = jnp.stack((r, -r))
        sign, log = vmap(self.logsignpsi, in_axes=(None, 0, 0))(params, r_pm, sz_pm)
        log = log + 1j * sign
        p_sign = jnp.asarray([1, -1])
        log = jax.scipy.special.logsumexp(a = log, b = p_sign, return_sign = False)
        return log

    @partial(jit, static_argnums=(0,))
    def psi(self, params, r, sz):
        log = self.logpsi( params, r, sz )
        return jnp.exp(log)

    @partial(jit, static_argnums=(0,))
    def phi_chi_h(self, phi_p_params, phi_a_params, rho_p_params, rho_a_params, x_i):
        """ Many-particle  radial functions 
        """
#        phi_p_h = jnp.mean(self.phi_apply(phi_p_params, x_i), axis=0)
#        phi_a_h = jnp.mean(self.phi_apply(phi_a_params, x_i), axis=0)

        phi_p_h = jax.scipy.special.logsumexp(self.phi_apply(phi_p_params, x_i), axis=0) 
        phi_a_h = jax.scipy.special.logsumexp(self.phi_apply(phi_a_params, x_i), axis=0) 

        rho_p_h = self.rho_apply(rho_p_params, phi_p_h)
        rho_a_h = self.rho_apply(rho_a_params, phi_a_h)

        rho_h = jnp.exp( self.a * jnp.tanh( rho_a_h / self.a ) + 1j * rho_p_h )

        return rho_h

    def vmap_psi(self, params, r_batched, sz_batched):
        return vmap(self.psi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)

    def psi_pmap(self, params, r_pmap, sz_pmap):
        return pmap(self.vmap_psi, in_axes=(None, 0, 0))(params, r_pmap, sz_pmap)
  
    @partial(jit, static_argnums=(0,))
    def vmap_logpsi(self, params, r_batched, sz_batched):
        return vmap(self.logpsi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)

    @partial(jit, static_argnums=(0,))
    def dlogpsi(self, params, r, sz):
        dlogpsi_dx = jax.grad(self.logpsi, argnums=1)(params, r, sz)
        return dlogpsi_dx

    @partial(jit, static_argnums=(0,))
    def vmap_dlogpsi(self, params, r_batched, sz_batched):
        return vmap(self.dlogpsi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)

    @partial(jit, static_argnums=(0,))
    def vmap_psi_sum(self, params, r_batched, sz_batched):
        return jnp.sum(self.vmap_psi(params, r_batched, sz_batched))

    @partial(jit, static_argnums=(0,))
    def flatten_params(self, parameters):
        flatten_parameters, self.unravel = ravel_pytree(parameters)
        return flatten_parameters 

    @partial(jit, static_argnums=(0,))
    def unflatten_params(self, flatten_parameters):
        unflatten_parameters = self.unravel(flatten_parameters)
        return unflatten_parameters 

    @partial(jit, static_argnums=(0,))
    def update_add(self, params, dparams):
        return params + dparams

    @partial(jit, static_argnums=(0,))
    def update_subtract(self, params, dparams):
        return params - dparams

    @partial(jit, static_argnums=(0,))
    def update_mix(self, params, dparams):
        return self.mix * params + (1 - self.mix) * dparams

    @partial(jit, static_argnums=(0,))
    def update_cut(self, params):
        return  4. * jnp.tanh( params / 4. )

    @partial(jit, static_argnums=(0,))
    def update_cast(self, params):
        return params.astype(jnp.float64)

    @partial(jit, static_argnums=(0,))
    def update_cast_complex(self, params):
        return params.astype(jnp.complex128)

    @partial(jit, static_argnums=(0,))
    def update_zero(self, params):
        return 0 * params

    @partial(jit, static_argnums=(0,))
    def pytrees_stack(self, pytrees, axis=0):
        results = jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
        return results
