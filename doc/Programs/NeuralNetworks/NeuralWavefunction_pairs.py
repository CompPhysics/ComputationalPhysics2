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
    act = 0.95
    return act * jnp.tanh(x) + (1 - act) * x 

LeakyTanh = elementwise(leakytanh)
Sin = elementwise(jnp.sin)

class Wavefunction(object):
    def __init__(self, ndim, npart, conf, key, mix, spin, remove_cm):
        self.ndim = ndim
        self.npart = npart
        self.conf = conf
        self.key = key
        self.mix = mix
        self.spin = spin
        self.remove_cm = remove_cm
        self.ip = spin.ip
        self.jp = spin.jp
        self.npair = spin.npair
        self.k = jnp.arange(self.npair)
        self.ndense = 16
        self.nsingle = 8 
        self.nlat = 16
        self.activation = LeakyTanh
        self.a = 8
        self.nhid = 3
        self.spin.nhid = self.nhid
        self.R0 = 1.2 * jnp.cbrt(self.npart) / jnp.sqrt( 3 )
        self.norm = 2

    def build(self):
# Deep-Sets: phi
        self.phi_init, self.phi_apply = stax.serial(
        Dense(self.ndense, b_init=zeros), self.activation,
        Dense(self.ndense, b_init=zeros), self.activation,
        Dense(self.nlat, b_init=zeros),
)

# Deep-Sets: rho visible
        rho_v_ff = stax.serial(
        Dense(self.ndense, b_init=zeros), self.activation,
        Dense(self.ndense, b_init=zeros), self.activation,
        Dense(self.npart, b_init=zeros),
)

# Deep-Sets: rho hidden
        rho_h_ff = stax.serial(
        Dense(self.ndense, b_init=zeros), self.activation,
        Dense(self.ndense, b_init=zeros), self.activation,
        Dense(self.nhid, b_init=zeros),
)

# Deep-Sets: combined rho hidden
        self.rho_init, self.rho_apply = stax.serial(
        FanOut(2),
        stax.parallel(rho_v_ff, rho_h_ff),
        FanInConcat()
)

# Feed-forward constructor for single-particle orbitals wave function  
        self.Rnl_init, self.Rnl_apply = stax.serial(
        Dense(self.nsingle, b_init=zeros), self.activation,
        Dense(self.nsingle, b_init=zeros), self.activation,
        Dense(1, b_init=zeros),
)

# Create self.nhid parallel structures of rho(phi(x))
        x_in_shape = (-1, self.ndim + 2)
        x_ij_in_shape = (-1, 2 * ( self.ndim + 2 ) ) 
        rho_in_shape = (-1, self.nlat ) 

        phi_p_params=[]
        phi_a_params=[]
        rho_p_params=[]
        rho_a_params=[]
        Rnl_h_params=[]
        for i in range(self.nhid):
            self.key, key_input = jax.random.split(self.key)
            _, phi_p_params_i = self.phi_init(key_input, x_ij_in_shape) 
            phi_p_params.append(phi_p_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, phi_a_params_i = self.phi_init(key_input, x_ij_in_shape) 
            phi_a_params.append(phi_a_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, rho_p_params_i = self.rho_init(key_input, rho_in_shape) 
            rho_p_params.append(rho_p_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, rho_a_params_i = self.rho_init(key_input, rho_in_shape) 
            rho_a_params.append(rho_a_params_i)

            self.key, key_input = jax.random.split(self.key)
            _, Rnl_h_params_i = self.Rnl_init(key_input, x_in_shape) 
            Rnl_h_params.append(Rnl_h_params_i)

        Rnl_v_params=[]
        for i in range(self.npart):
            self.key, key_input = jax.random.split(self.key)
            _, Rnl_v_params_i = self.Rnl_init(key_input, x_in_shape) 
            Rnl_v_params.append(Rnl_v_params_i)

        self.num_phi_p_params = len(phi_p_params)
        self.num_phi_a_params = len(phi_a_params)
        self.num_rho_p_params = len(rho_p_params)
        self.num_rho_a_params = len(rho_a_params)
        self.num_Rnl_h_params = len(Rnl_h_params)
        self.num_Rnl_v_params = len(Rnl_v_params)

        net_params = phi_p_params + phi_a_params + rho_p_params + rho_a_params + Rnl_v_params + Rnl_h_params

# cast to double
        net_params = jax.tree_util.tree_map(self.update_cast, net_params)
        flat_net_params = self.flatten_params(net_params)
        num_flat_params = flat_net_params.shape[0]

        with open('full.model', 'wb') as file:
             pickle.dump(net_params, file)

        return net_params, num_flat_params

    @partial(jit, static_argnums=(0,))
    def logsignpsi_pm(self, params, r, sz):

        num_offset_params = 0

        phi_p_params = params[num_offset_params : num_offset_params + self.num_phi_p_params]
        num_offset_params = num_offset_params + self.num_phi_p_params

        phi_a_params = params[num_offset_params : num_offset_params + self.num_phi_a_params]
        num_offset_params = num_offset_params + self.num_phi_a_params

        rho_p_params = params[num_offset_params : num_offset_params + self.num_rho_p_params]
        num_offset_params = num_offset_params + self.num_rho_p_params

        rho_a_params = params[num_offset_params : num_offset_params + self.num_rho_a_params]
        num_offset_params = num_offset_params + self.num_rho_a_params

        Rnl_v_params = params[num_offset_params : num_offset_params + self.num_Rnl_v_params]
        num_offset_params = num_offset_params + self.num_Rnl_v_params

        Rnl_h_params = params[num_offset_params : num_offset_params + self.num_Rnl_h_params]
        num_offset_params = num_offset_params + self.num_Rnl_h_params

        if (self.remove_cm):
           rcm = jnp.mean(r, axis=0)
           r = ( r - rcm[None,:] ) 
        r = r / self.R0
        x = jnp.concatenate((r, sz ), axis=1)

        def x_pair(k, x):
            x_ij = jnp.concatenate((x[self.ip[k],:],x[self.jp[k],:]))
            x_ji = jnp.concatenate((x[self.jp[k],:],x[self.ip[k],:]))
            return x_ij, x_ji
        x_ij, x_ji = vmap(x_pair, in_axes=(0, None)) (self.k, x) 
        x_ij = jnp.append(x_ij, x_ji, axis=0)

# rho_h is (nhid, npart + nhid ) where I concatenate rho(npart) and rho(nhid)
        phi_p_params_stacked = self.pytrees_stack(phi_p_params)
        phi_a_params_stacked = self.pytrees_stack(phi_a_params)
        rho_p_params_stacked = self.pytrees_stack(rho_p_params)
        rho_a_params_stacked = self.pytrees_stack(rho_a_params)
        rho_h = vmap(self.phi_chi_h, in_axes=(0, 0, 0, 0, None))(phi_p_params_stacked, phi_a_params_stacked, rho_p_params_stacked, rho_a_params_stacked, x_ij)

# Rnl_v is (nstate=npart, npart)
        Rnl_v_params_stacked = self.pytrees_stack(Rnl_v_params)
        Rnl_v = self.norm * jnp.reshape(vmap(self.Rnl_apply, in_axes=(0, None))(Rnl_v_params_stacked, x), newshape=[self.npart, self.npart])

# Rnl_h is (nstate=nhid, npart)
        Rnl_h_params_stacked = self.pytrees_stack(Rnl_h_params)
        Rnl_h = self.norm * jnp.reshape(vmap(self.Rnl_apply, in_axes=(0, None))(Rnl_h_params_stacked, x), newshape=[self.nhid, self.npart])

        sign, logdet = self.spin.phi_spin(sz, Rnl_v, Rnl_h, rho_h)
        log = logdet -  self.conf * jnp.sum( r**2 )

        sign = jnp.reshape(sign, ())
        log = jnp.reshape(log, ())
        return sign, log  

    @partial(jit, static_argnums=(0,))
    def logsignpsi(self, params, r, sz):
        sz_m = sz.at[:,0].set(-sz[:,0])
#        sz_pm = jnp.stack((sz, sz_m, sz, sz_m))
#        r_pm = jnp.stack((r, r, -r, -r))
        sz_pm = jnp.stack((sz, sz))
        r_pm = jnp.stack((r, -r))
        sign, log = vmap(self.logsignpsi_pm, in_axes=(None, 0, 0))(params, r_pm, sz_pm) 
        log_pm, sign_pm = jax.scipy.special.logsumexp(a = log, b = sign, return_sign=True)

#        sign_pm, log_pm = self.logsignpsi_pm(params, r, sz)
        return sign_pm, log_pm

    @partial(jit, static_argnums=(0,))
    def psi(self, params, r, sz):
        sign, log = self.logsignpsi( params, r, sz )
        psi = sign * jnp.exp(log) 
        return psi  

    @partial(jit, static_argnums=(0,))
    def logpsi(self, params, r, sz):
        _, log = self.logsignpsi( params, r, sz )
        return log  

    @partial(jit, static_argnums=(0,))
    def signpsi(self, params, r, sz):
        sign, _ = self.logsignpsi( params, r, sz )
        return sign  
        
    @partial(jit, static_argnums=(0,))
    def phi_chi_h(self, phi_p_params, phi_a_params, rho_p_params, rho_a_params, x_i):
        """ Many-particle  radial functions 
        """
        phi_p_h = jnp.mean(self.phi_apply(phi_p_params, x_i), axis=0)
        phi_a_h = jnp.mean(self.phi_apply(phi_a_params, x_i), axis=0)

#        phi_p_h = jax.scipy.special.logsumexp(self.phi_apply(phi_p_params, x_i), axis=0) 
#        phi_a_h = jax.scipy.special.logsumexp(self.phi_apply(phi_a_params, x_i), axis=0) 

        rho_p_h = jnp.tanh(self.rho_apply(rho_p_params, phi_p_h))
        rho_a_h = jnp.exp(self.rho_apply(rho_a_params, phi_a_h))

        return self.norm * rho_p_h * rho_a_h 

    @partial(jit, static_argnums=(0,))
    def Y_1m_sp(self, r_i):
        norm = jnp.sqrt(3 / 4 / jnp.pi)
        Y_1m = jnp.zeros(3)
        Y_1m = Y_1m.at[0].set( norm * r_i[1] )
        Y_1m = Y_1m.at[1].set( norm * r_i[2] )
        Y_1m = Y_1m.at[2].set( norm * r_i[0] )
        return Y_1m
  
    @partial(jit, static_argnums=(0,))
    def vmap_psi(self, params, r_batched, sz_batched):
        return vmap(self.psi, in_axes=(None, 0, 0))(params, r_batched, sz_batched)  

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
    def update_zero(self, params):
        return 0 * params

    @partial(jit, static_argnums=(0,))
    def pytrees_stack(self, pytrees, axis=0):
        results = jax.tree_util.tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
        return results
