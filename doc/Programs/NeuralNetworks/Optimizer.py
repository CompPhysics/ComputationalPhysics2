import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap, pmap, jacfwd, jacrev
from jax.lax import fori_loop, psum, pmean
from jax.scipy.linalg import cho_factor, cho_solve
from functools import partial

import logging
logger = logging.getLogger()


@jit
def conj_transpose(M):
    return jnp.conjugate(jnp.transpose(M))


class Optimizer(object):

    def __init__(self, n_devices, delta, eps, nparams, nwalk, npart, ndim, nav, nac, wavefunction, observables, spin, solver, remove_cm):
        self.lbd = 0.0
        self.alpha = 0.9
        self.beta = 0.99
        self.n_devices = n_devices
        self.eps = eps
        self.delta = delta
        self.nparams = nparams
        self.nwalk = nwalk
        self.npart = npart
        self.ndim = ndim
        self.nav = nav 
        self.nac = nac
        self.wavefunction = wavefunction
        self.observables = observables
        self.spin = spin
        self.solver = solver
        self.remove_cm = remove_cm
        self.itr = 0
        self.dp_i = jnp.zeros(self.nparams)
        self.g2_i = jnp.zeros(self.nparams)
        self.m_i = jnp.zeros(self.nparams)
        self.f_i = jnp.zeros(self.nparams)
        self.nstep = 1
        self.solve_cg_eps = jax.jit(self.solve_cg_eps)
        self.vmap_getder = jax.jit(self.vmap_getder)
        self.getder = jax.jit(self.getder)
        self.dist = jax.jit(self.dist)

    def getder(self, params, r, sz):
        params = jax.tree_util.tree_map(self.wavefunction.update_cast_complex, params)
        logpsi = lambda params: self.wavefunction.logpsi(params, r, sz)
        dlogpsi = jax.grad(logpsi, holomorphic = True)(params)
        dlogpsi = self.wavefunction.flatten_params(dlogpsi)
        return dlogpsi

    def pmap_getder(self, params, r, sz, nsamples):
        jac_pmap = pmap(self.vmap_getder, in_axes=(None, 0, 0, None), axis_name='p')(params, r, sz, nsamples)
        jac_pmap = jnp.reshape(jac_pmap, (-1, self.nparams))
        return jac_pmap

    def vmap_getder(self, params, r, sz, nsamples):
        jac = vmap(self.getder, in_axes=(None, 0, 0), out_axes=(0))(params, r, sz)
        jac = jac - psum(jnp.sum(jac, axis=0), axis_name='p') / nsamples
        return jac

    def solve_cg_eps(self, params, r, sz, energy, aux):
      """ Solves the SR with the CG method. 

      r, sz, and energy are subsets of size nsamples / ndevices
      The full f_i g2_i, and g2h_i are computed exactly the same in each device.
      Inefficient in energy but efficient in memory and time.
      """

      dp0_i, g2_i, itr, nsamples = aux

      jac = self.vmap_getder(params, r, sz, nsamples)
      #jac_T = conj_transpose(jac)

      f_i = jnp.real(-2 * psum(jnp.matmul(energy, jnp.conjugate(jac)), axis_name='p') / nsamples)
      g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2
      g2h_i = jnp.sqrt(g2_i / (1. - self.beta**itr))

#      cg_mult = lambda v_i: jnp.real(psum(jnp.matmul(jnp.matmul(v_i, jac_T), jac), axis_name='p') / nsamples) + self.eps * (0.001 + g2h_i) * v_i
      cg_mult = lambda v_i: jnp.real(psum(jnp.matmul(jnp.matmul(jnp.conjugate(jac), v_i), jac), axis_name='p') / nsamples) + self.eps * (0.001 + g2h_i) * v_i

      dp_i, info = jax.scipy.sparse.linalg.cg(cg_mult, f_i, x0=dp0_i, tol=1e-5, atol=0.0, maxiter=200)
      return dp_i, g2_i

    def sr_cg(self, params, r, sz, energy, dp0_i, g2_i, itr):
      """Parameters' update according to the SR algorithm with Conjugate Gradient solver
      Args:
      params: initial variational parameters
      r: array with shape (n_devices, nwalk * nav, npart, ndim)
      sz: array with shape (n_devices, nwalk * nav, npart, 2)
      energy: array with shape (n_devices, nwalk, nav) 
      dp0_i: array with shape (nparams) representing the initial guess of the CG solver
      g2_i: accumulated second order derivative squared  (nparams) 
      itr : iteration

      Returns:
      dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
      g2_i: updated accumulated second order derivative squared (nparams)
      """

      energy = energy - jnp.mean(energy)
      nsamples = self.nwalk * self.nav
      aux = (dp0_i, g2_i, itr, nsamples)

      dp_i, g2_i = pmap(self.solve_cg_eps, in_axes=(None, 0, 0, 0, None), axis_name='p')(params, r, sz, energy, aux)
      dp_i = jnp.mean(dp_i[:], axis = 0)
      g2_i = jnp.mean(g2_i[:], axis = 0)

      dp_i = dp_i - self.lbd * self.wavefunction.flatten_params(params)
      return dp_i, g2_i

    def sr_pseudo(self, params, r, sz, energy):
      """Parameters' update according to the SR algorithm with Pseudo-Inverse
      Args:
      params: initial variational parameters
      r: array with shape (n_devices, nwalk * nav, npart, ndim)
      sz: array with shape (n_devices, nwalk * nav, npart, 2)
      energy: array with shape (n_devices, nwalk, nav) 

      Returns:
      dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
      """

      energy = energy - jnp.mean(energy)
      nsamples = self.nwalk * self.nav
      energy = jnp.reshape(energy, (nsamples))

      jac = self.pmap_getder(params, r, sz, nsamples)
      jac_T = jnp.transpose(jac)
      F = jnp.matmul(jac, jac_T) 
      shift = nsamples * jnp.identity(nsamples)
      dp_i = jnp.linalg.solve(F + self.eps * shift, energy)
      dp_i = - 2 * jnp.matmul(jac_T, dp_i) - self.lbd * self.wavefunction.flatten_params(params)
      return dp_i

    def adam(self, params, r, sz, energy, g2_i, m_i, itr):
      """Parameters' update according to the Adam algorithm
      Args:
      params: initial variational parameters
      r: array with shape (n_devices, nwalk * nav, npart, ndim)
      sz: array with shape (n_devices, nwalk * nav, npart, 2)
      energy: array with shape (n_devices, nwalk * nav) 
      m_i: accumulated momentum  (nparams) 
      g2_i: accumulated second order derivative squared  (nparams) 
      itr : iteration

      Returns:
      dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
      g2_i: updated accumulated second order derivative squared (nparams)
      m_i: updated accumulated momentum (nparams)
      """

      energy = energy - jnp.mean(energy)

      nsamples = self.nwalk * self.nav
      energy = jnp.reshape(energy, (nsamples))
      jac = self.pmap_getder(params, r, sz, nsamples)

      f_i = - 2 * jnp.matmul(energy, jac) / nsamples 
      m_i = self.alpha * m_i + (1. - self.alpha) * f_i
      mh_i = m_i / ( 1. - self.alpha**itr )
      g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2 
      g2h_i = g2_i / ( 1. - self.beta**itr )
      dp_i = mh_i / ( jnp.sqrt(g2h_i) + 0.00001 ) - self.lbd * self.wavefunction.flatten_params(params)  
      return dp_i, g2_i, m_i

    def sr_cholesky(self, params, r, sz, energy, g2_i, itr):
      """Parameters' update according to the SR algorithm with Cholesky solver
      Args:
      params: initial variational parameters
      r: array with shape (n_devices, nwalk * nav, npart, ndim)
      sz: array with shape (n_devices, nwalk * nav, npart, 2)
      energy: array with shape (n_devices, nwalk, nav) 
      g2_i: accumulated second order derivative squared  (nparams) 
      itr : iteration

      Returns:
      dp_i: array with the same shape as ``params`` representing the best parameters' update (nparams)
      g2_i: updated accumulated second order derivative squared (nparams)
      """

      energy = energy - jnp.mean(energy)

      nsamples = self.nwalk * self.nav
      energy = jnp.reshape(energy, (nsamples))
      jac = self.pmap_getder(params, r, sz, nsamples)

      f_i = - 2 * jnp.matmul(energy, jac) / nsamples 
      g2_i = self.beta * g2_i + (1. - self.beta) * f_i**2 
      g2h_i = g2_i / ( 1. - self.beta**itr )
      S_ij = jnp.matmul(jnp.transpose(jac), jac) / nsamples + self.eps * jnp.diag( 0.001 + jnp.sqrt(g2h_i) )

      U_ij, low = cho_factor(S_ij)
      dp_i = cho_solve((U_ij, low), f_i)
      return dp_i, g2_i 
       
# Solve the SR equations 
    def sr(self, params, r_s, sz_s, energy_s):
      """Optimizes the wave function according to different training algorithm

      Args:
      params: initial variational parameters
      r_s: array with shape (n_devices, nwalk, nav + nac, npart, ndim)
      sz: array with shape (n_devices, nwalk, nav + nac, npart, 2)
      energy_s: array with shape (n_devices, nwalk, nav + nac) 

      Returns:
      Array with the same shape as ``params`` representing the best parameters' update
      """

      self.itr += 1

      r_av = jnp.reshape(r_s[:,:,0:self.nav,:,:], (self.n_devices, self.nwalk // self.n_devices * self.nav, self.npart, self.ndim))
      sz_av = jnp.reshape(sz_s[:,:,0:self.nav,:,:], (self.n_devices, self.nwalk // self.n_devices * self.nav, self.npart, 2))
      energy_av = jnp.reshape(energy_s[:,:,0:self.nav], (self.n_devices, self.nwalk // self.n_devices * self.nav))

      r_val = jnp.reshape(r_s[:,:,0:self.nac,:,:], (self.n_devices, self.nwalk // self.n_devices * self.nac, self.npart, self.ndim))
      sz_val = jnp.reshape(sz_s[:,:,0:self.nac,:,:], (self.n_devices, self.nwalk // self.n_devices * self.nac, self.npart, 2))
      energy_val = jnp.reshape(energy_s[:,:,0:self.nac], (self.n_devices, self.nwalk // self.n_devices * self.nac))
      r_avg, r_max = self.observables.radius_pmap(r_val, sz_val)
      logger.info(f"Maximum Radius validation=, {jnp.sqrt(r_max):.3f}  ")
      logger.info(f"Average Radius validation=, {jnp.sqrt(r_avg):.3f}  ")
      psi_val = self.wavefunction.psi_pmap(params, r_val, sz_val)
      logger.info(f"Maximum Psi validation= ,{jnp.max(jnp.abs(psi_val)):.4e}  ")
      logger.info(f"Average Psi validation= ,{jnp.mean(jnp.abs(psi_val)):.4e}  ")
      logger.info(f"Minimum Psi validation= ,{jnp.min(jnp.abs(psi_val)):.4e}  ")

      r_tst = jnp.reshape(r_s[:,:,self.nav:self.nav+self.nac,:,:], (self.n_devices, self.nwalk // self.n_devices * self.nac, self.npart, self.ndim))
      sz_tst = jnp.reshape(sz_s[:,:,self.nav:self.nav+self.nac,:,:], (self.n_devices, self.nwalk // self.n_devices * self.nac, self.npart, 2))
      energy_tst = jnp.reshape(energy_s[:,:,self.nav:self.nav+self.nac], (self.n_devices, self.nwalk // self.n_devices * self.nac))
      r_avg, r_max = self.observables.radius_pmap(r_tst, sz_tst)
      logger.info(f"Maximum Radius test=, {jnp.sqrt(r_max):.3f}  ")
      logger.info(f"Average Radius test=, {jnp.sqrt(r_avg):.3f}  ")
      psi_tst = self.wavefunction.psi_pmap(params, r_tst, sz_tst)
      logger.info(f"Maximum Psi test= ,{jnp.max(jnp.abs(psi_tst)):.4e}  ")
      logger.info(f"Average Psi test= ,{jnp.mean(jnp.abs(psi_tst)):.4e}  ")
      logger.info(f"Minimum Psi test= ,{jnp.min(jnp.abs(psi_tst)):.4e}  ")

      if (self.solver == 'Adam'):
         dp_i, self.g2_i, self.m_i = self.adam(params, r_av, sz_av, energy_av, g2_i = self.g2_i, m_i = self.m_i, itr = self.itr)
      elif (self.solver == 'Cholesky'):
         dp_i, self.g2_i = self.sr_cholesky(params, r_av, sz_av, energy_av, g2_i = self.g2_i, itr = self.itr)  
      elif (self.solver == 'Pseudo'):
         dp_i = self.sr_pseudo(params, r_av, sz_av, energy_av) 
      elif (self.solver == 'CG'):
         dp_i, self.g2_i = self.sr_cg(params, r_av, sz_av, energy_av, dp0_i=self.dp_i, g2_i = self.g2_i, itr = self.itr) 

      self.dp_i = dp_i
      energy_d_min= 1.
      converged = False
      dp_range = jnp.linspace(0.1, 0.8, self.nstep)
      for n in range (self.nstep):
#          dt = dt_range[n]
          dp_n = self.delta * dp_i#[n] #+ dp_range[n] * self.dp_o
          dp_max = jnp.max(jnp.abs(dp_n))    
          delta_p = self.wavefunction.unflatten_params(dp_n) 
#          delta_p = self.wavefunction.unflatten_params( self.delta * dp_i ) 

          psi_d_tst, psi_d_err_tst, energy_d_tst, energy_d_err_tst, overlap_tst = self.pmap_dist(delta_p, params, r_tst, sz_tst, psi_tst, energy_tst)
          dist_tst = jnp.arccos(jnp.sqrt(psi_d_tst))**2
          logger.debug(f"dist acos tst = {dist_tst:.8f}")
          logger.debug(f"energy diff tst = {energy_d_tst:.6f}, err = {energy_d_err_tst:.6f}")
          logger.debug(f"overlap tst= { jnp.arccos(jnp.sqrt(overlap_tst))**2:.8f}")
  
          psi_d_val, psi_d_err_val, energy_d_val, energy_d_err_val, overlap_val = self.pmap_dist(delta_p, params, r_val, sz_val, psi_val, energy_val)
          dist_val = jnp.arccos(jnp.sqrt(psi_d_val))**2
          logger.debug(f"dist acos val = {dist_val:.8f}")
          logger.debug(f"energy diff val= {energy_d_val:.6f}, err = {energy_d_err_val:.6f}")
          logger.debug(f"overlap val = { jnp.arccos(jnp.sqrt(overlap_val))**2:.8f}")

          logger.debug(f"delta param max = {dp_max:.6f}")
          logger.debug(f"delta param avg = {jnp.linalg.norm(dp_n):.6f}")

          if ( dist_tst < 0.1 and dist_val < 0.1 and energy_d_val < energy_d_min and dp_max < 0.5):
             energy_d_min = energy_d_val
             energy_d_err_min = energy_d_err_val
             delta_p_min = delta_p
             dp_n_min = dp_n
             converged = True
      if converged:       
         logger.debug(f"Converged, energy diff min = {energy_d_min:.6f}, err = {energy_d_err_min:.6f}")
      else:
         logger.debug(f"Not converged")
         delta_p_min = self.wavefunction.unflatten_params(jnp.zeros(self.nparams))
         dp_n_min = jnp.zeros(self.nparams)
      return delta_p_min

    def pmap_dist(self, delta_p, params, r, sz, psi, energy):
        nsamples = self.nwalk * self.nac
        psi_d_sum, psi_d_err, energy_d_sum, energy_d_err, overlap = pmap(self.dist, in_axes=(None, None, None, 0, 0, 0, 0), axis_name='p')(delta_p, params, nsamples, r, sz, psi, energy)
        return jnp.mean(psi_d_sum[:]), jnp.mean(psi_d_err[:]), jnp.mean(energy_d_sum[:]), jnp.mean(energy_d_err[:]), jnp.mean(overlap[:])
        
    def dist(self, delta_p, params, nsamples, r, sz, psi_o, energy_o):

# Compute the old wave function and energy
        energy_o_sum = psum(jnp.sum(energy_o), axis_name='p') / nsamples 
        energy2_o_sum = psum(jnp.sum(energy_o**2), axis_name='p') / nsamples 
        energy_o_err = jnp.sqrt((energy2_o_sum - energy_o_sum**2) / nsamples)

# Update the parameters
        params_n = jax.tree_util.tree_map(self.wavefunction.update_add, params, delta_p)

# Compute the new wave function and energy reweighting the stored walk
        psi_n = self.wavefunction.vmap_psi(params_n, r, sz)
        obs_n = self.observables.energy(params_n, r, sz)
        energy_n = obs_n[:,0]
        psi_ratio = psi_n / psi_o
        psi2_norm_sum = psum(jnp.sum(psi_ratio**2), axis_name='p') / nsamples
        psi_norm_sum = psum(jnp.sum(psi_ratio), axis_name='p') / nsamples

        energy_n *= psi_ratio**2
        energy_n_sum = psum( jnp.sum(energy_n), axis_name='p') / nsamples / psi2_norm_sum
        energy2_n_sum = psum( jnp.sum(energy_n**2), axis_name='p') / nsamples / psi2_norm_sum
        energy_n_err = jnp.sqrt((energy2_n_sum - energy_n_sum**2) / nsamples )

# Correlated energy difference
        energy_d = energy_n / psi2_norm_sum - energy_o
        energy_d_sum = psum(jnp.sum(energy_d), axis_name='p') / nsamples
        energy2_d_sum = psum(jnp.sum(energy_d**2), axis_name='p') / nsamples
        energy_d_err = jnp.sqrt((energy2_d_sum - energy_d_sum**2) / nsamples )

# Overlap with the previous wave function <Psi_{p+dp}|Psi_p><Psi_p|Psi_{p+dp}> / <Psi_{p+dp}|Psi_{p+dp}> / <Psi_p|Psi_p> 
        psi_d_sum = psi_norm_sum**2 / psi2_norm_sum
        psi_d_err = jnp.sqrt( (psi2_norm_sum - psi_norm_sum**2) / nsamples)

# Overlap between (1 - H dt)|Psi_p> and |Psi_{p+dp}> using samples from |Psi_p(x)|^2

        psi_dt = (1. - energy_o * self.delta) 
        psi_n = psi_n / psi_o
        overlap = psum(jnp.sum(psi_n * psi_dt), axis_name='p') / nsamples
        norm_n = psum(jnp.sum(psi_n * psi_n), axis_name='p') / nsamples
        norm_dt = psum(jnp.sum(psi_dt * psi_dt), axis_name='p') / nsamples
        overlap = overlap**2 / norm_n / norm_dt 
        return psi_d_sum, psi_d_err, energy_d_sum, energy_d_err, overlap

#    @partial(jit, static_argnums=(0,))
    def overlap_compute(self, delta_p, params, r, sz):

        wpsi_o = self.wavefunction.psi_pmap(params, r, sz)
        obs_o = self.observables.energy_pmap(params, r, sz)
        energy_o = obs_o[:,:,0]

# the wave function at t+dt is |psi(t+dt)> = (1 - H * dt ) |psi(t)>, but we need to divide by wpsi_o to normalize
        wpsi_dt = (1. - energy_o * self.delta) 

        params_n = jax.tree_util.tree_map(self.wavefunction.update_add, params, delta_p)
        wpsi_n = self.wavefunction.psi_pmap(params_n, r, sz) / wpsi_o

        overlap = jnp.mean(wpsi_n * wpsi_dt)**2
        norm_n = jnp.mean( jnp.abs(wpsi_n)**2)
        norm_dt = jnp.mean( jnp.abs(wpsi_dt)**2)
        overlap = overlap / norm_n / norm_dt 

# Set back the old parameters
#        params = jax.tree_util.tree_map(self.wavefunction.update_subtract, params, delta_p)

        return overlap

    def split(self, arr):
        return arr.reshape(self.n_devices, arr.shape[0] // self.n_devices, *arr.shape[1:])
