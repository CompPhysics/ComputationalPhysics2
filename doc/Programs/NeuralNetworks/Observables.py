import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import fori_loop, psum, pmean
from jax import random, grad, jit, vmap, jacfwd, jacrev, pmap 
from jax.lax import fori_loop
from functools import partial

class Observables(object):
    """Observables class
    """

    def __init__(self, n_devices, mass, hbar, omega, ndim, npart, wavefunction, spin, potential, remove_cm):
        self.ndim = ndim
        if self.ndim < 1 or self.ndim > 3: 
            raise Exception("Dimension must be 1, 2, or 3")

        self.n_devices = n_devices
        self.mass = mass
        self.hbar = hbar
        self.omega = omega
        self.hbar2m = self.hbar**2 / self.mass / 2.
        self.npart = npart
        self.wavefunction = wavefunction
        self.spin = spin
        self.npair = spin.npair
        self.potential = potential
        self.ip = spin.ip
        self.jp = spin.jp
        self.remove_cm = remove_cm

        self.basis = jnp.zeros(shape=[self.npart * self.ndim, self.npart * self.ndim])
        for i in range(self.npart * self.ndim):
            self.basis = self.basis.at[i,i].set(1)
        self.basis = jnp.reshape(self.basis, (self.ndim * self.npart, self.npart, self.ndim))

        # Several objects get stored for referencing, if needed, after energy computation:
        self.nrho = 100
        self.rho_min = 0.
        self.rho_max= 10.
        self.r_rho = np.linspace(self.rho_min, self.rho_max, num=self.nrho+1)
        self.v_rho = np.zeros(shape = [self.nrho])
        for i in range (self.nrho):
            self.v_rho[i] = self.r_rho[i+1]**3 - self.r_rho[i]**3
        self.v_rho = 4./3. * jnp.pi * self.v_rho
        self.energy=jax.jit(self.energy)
        self.radius=jax.jit(self.radius)



    @partial(jax.jit, static_argnums=(0,))
    def v_pair(self, k, params, r, sz, logpsi):
        r_ij = r[self.ip[k],:] - r[self.jp[k],:]
        r_ij = jnp.sqrt(jnp.sum(r_ij**2))
        vr_ij = self.potential.v_2b(r_ij)
        vrem_ij = self.potential.v_em(r_ij)

        vc_ij = vr_ij[0]
        sz_ij = self.spin.sp_exch(sz, k, 0)
        tz_ij = self.spin.sp_exch(sz, k, 1)
        stz_ij = self.spin.sp_exch(tz_ij, k, 0)

        logpsi_t_ij  = self.wavefunction.logpsi(params, r, tz_ij)
        Pt_ij = jnp.exp(logpsi_t_ij - logpsi)

        logpsi_s_ij  = self.wavefunction.logpsi(params, r, sz_ij)
        Ps_ij = jnp.exp(logpsi_s_ij - logpsi)

        logpsi_st_ij  = self.wavefunction.logpsi(params, r, stz_ij)
        Pst_ij = jnp.exp(logpsi_st_ij - logpsi)

        vt_ij = vr_ij[1] * ( 2 * Pt_ij - 1 )
        vs_ij = vr_ij[2] * ( 2 * Ps_ij - 1 )
        vst_ij = vr_ij[3] * ( 4 * Pst_ij - 2 * Pt_ij - 2 * Ps_ij + 1 )
        vem_ij = ( 1 + sz[self.ip[k],1] ) * ( 1 + sz[self.jp[k],1] ) / 4 * vrem_ij

        t_ij = self.potential.v_3b(r_ij)
        return vc_ij, vt_ij, vs_ij, vst_ij, vem_ij, t_ij  

    @partial(jax.jit, static_argnums=(0,))
    def potential_energy(self, params, r, sz):
        "Returns potential energy"
        "Returns potential energy"

        v_ij = jnp.zeros(6, dtype=jnp.complex128)
        gr3b = jnp.zeros(self.npart, dtype=jnp.complex128)
        vem_ij = 0
        V_ijk = 0
        k = jnp.arange(self.npair)

        logpsi = self.wavefunction.logpsi( params, r, sz )
        v_pair_map = lambda k: self.v_pair(k, params, r, sz, logpsi)
#        vc_ij, vt_ij, vs_ij, vst_ij, vem_ij, t_ij = jax.lax.map(vmap(v_pair_map), k.reshape((-1, self.npart - 1) + k.shape[1:]))
#        vc_ij, vt_ij, vs_ij, vst_ij, vem_ij, t_ij = vmap(v_pair_map)(k)
        vc_ij, vt_ij, vs_ij, vst_ij, vem_ij, t_ij = jax.lax.map(v_pair_map, k)

        vc_ij = vc_ij.reshape(self.npair)
        vt_ij = vt_ij.reshape(self.npair)
        vs_ij = vs_ij.reshape(self.npair)
        vst_ij = vst_ij.reshape(self.npair)
        vem_ij = vem_ij.reshape(self.npair)
        t_ij = t_ij.reshape(self.npair)

        v_ij = v_ij.at[0].add(jnp.sum(vc_ij[:]))
        v_ij = v_ij.at[1].add(jnp.sum(vt_ij[:]))
        v_ij = v_ij.at[2].add(jnp.sum(vs_ij[:]))
        v_ij = v_ij.at[3].add(jnp.sum(vst_ij[:]))

        if (self.npart > 2 ):
           for k in range (self.npair):
               gr3b = gr3b.at[self.ip[k]].add(t_ij[k])
               gr3b = gr3b.at[self.jp[k]].add(t_ij[k])
           V_ijk = 0.5 * jnp.sum(gr3b**2) - jnp.sum(t_ij**2)

        pe = v_ij[0] + v_ij[1] + v_ij[2] + v_ij[3] + jnp.sum(vem_ij) + V_ijk

#        if (self.vext):
#           vext = self.omega * jnp.sum(r**2)
#           pe += vext

        return pe

    @partial(jax.jit, static_argnums=(0,))
    def kinetic_energy(self, params, r, sz):
        "Returns kinetic energy"

        logpsi = lambda r: self.wavefunction.logpsi(params, r, sz)
        d_logpsi = jax.grad(logpsi, holomorphic = True)(r.astype(jnp.complex128))
        d2_logpsi = jax.hessian(logpsi, holomorphic = True)(r.astype(jnp.complex128))
        d2_logpsi = jnp.reshape(d2_logpsi,(self.ndim * self.npart, self.ndim * self.npart)) 
        ke = - self.hbar2m * ( jnp.trace(d2_logpsi) + jnp.sum( d_logpsi * d_logpsi ) ) 
        ke_jf = self.hbar2m * jnp.sum( jnp.conj(d_logpsi) * d_logpsi )

        return ke, ke_jf


    def jacrev(self, f):
        def jacfun(x):
            y, vjp_fun = jax.vjp(f, x)
            eye = jnp.eye(y.size)[0]
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
            return J
        return jacfun

    def jacfwd(self, f):
        def jacfun(x):
            jvp_fun = lambda s: jax.jvp(f, (x,), (s,))[1]
            eye = jnp.eye(len(x))
            J = jax.vmap(jvp_fun, in_axes=0)(eye)
            return J
        return jacfun

    @partial(jax.vmap, in_axes=(None, None, 0, 0))
    def kinetic_gabriel(self, params, x, sz):
        x = x.reshape((self.npart * self.ndim)) 
           
        def logpsi_x(x):
            x = x.reshape((self.npart, self.ndim)) 
            return self.wavefunction.logpsi(params, x, sz)

        dlogpsi_x = self.jacrev(logpsi_x)
        dp_dx2 = jnp.diag(self.jacfwd(dlogpsi_x)(x)[0].reshape(x.shape[0], x.shape[0]))
        dp_dx = dlogpsi_x(x)[0][0] ** 2
        return -self.hbar2m * jnp.sum(dp_dx2 + dp_dx, axis=-1)


    def hessian_diag(self, f, x):
        def hvp(f, x, v):
            return jax.jvp(jax.grad(f, holomorphic = True), [x], [v])[1]
        comp = lambda v: jnp.vdot(v, hvp(f, x, v))
        return jax.vmap(comp)(self.basis)

    def energy(self, params, r, sz):
        "Returns the toal energy vmapping over the input energy"
        ke, ke_jf = vmap(self.kinetic_energy, in_axes=(None, 0, 0), out_axes=(0))(params, r, sz) 
#        ke = self.kinetic_gabriel(params, r, sz)
        pe = vmap(self.potential_energy, in_axes=(None, 0, 0))(params, r, sz)
        energy_jf = ke_jf + pe
        energy = ke + pe
#        energy_jf = pe
#        energy = pe
#        energy = jnp.zeros(shape=r.shape[0])
#        energy_jf = jnp.zeros(shape=r.shape[0])

        obs = jnp.zeros(shape = [energy.shape[0],2], dtype=jnp.complex128)
        obs = obs.at[:,0].set(energy)
        obs = obs.at[:,1].set(energy_jf)
        return obs

    def energy_pmap(self, params, r_pmap, sz_pmap):
        "Returns the toal energy pmapping over the input energy"
        return pmap(self.energy, in_axes=(None, 0, 0))(params, r_pmap, sz_pmap)

    def density(self, r):
        """Computes the expectation value of the single-nucleon density
        """
        jnp.set_printoptions(threshold=np.inf)
        if (self.remove_cm):
           rcm = jnp.mean(r, axis=1)
           r = r - rcm[:,None,:]
        r = jnp.sqrt(jnp.sum(r**2,axis=(2)))
        rho = jnp.histogram(r, bins=self.nrho, range=(self.rho_min, self.rho_max)) 
        rho = rho[0] / self.v_rho
        return rho

    def density_print(self, rho, error_rho):
        jnp.set_printoptions(precision=8, threshold=np.inf)
        for i in range (self.nrho):
            print((self.r_rho[i]+self.r_rho[i+1])/2.,rho[i],error_rho[i])
        return

    def spin_isospin(self, sz):
        Sz = jnp.mean(sz[:,:,:,0]) * self.npart / 2 
        Tz = jnp.mean(sz[:,:,:,1]) * self.npart / 2
        return Sz, Tz

    def radius(self, r, sz):
        """Computes the expectation value of the single-nucleon radius
        """
        if (self.remove_cm):
           rcm = jnp.mean(r, axis=1)
           r = r - rcm[:,None,:]
        r2 = jnp.sum(r**2, axis = 2)        
        r_avg = jnp.mean(r2)
        r_max = jnp.max(r2)
        return r_avg, r_max

    def radius_pmap(self, r_pmap, sz_pmap):
        r_avg_pmap, r_max_pmap = pmap(self.radius, in_axes=(0, 0))(r_pmap, sz_pmap)
        return jnp.mean(r_avg_pmap), jnp.max(r_max_pmap)

    def split(self, arr):
        return arr.reshape(self.n_devices, arr.shape[0] // self.n_devices, *arr.shape[1:])

