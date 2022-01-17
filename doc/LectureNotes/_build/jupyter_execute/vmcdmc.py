#!/usr/bin/env python
# coding: utf-8

# <!-- HTML file automatically generated from DocOnce source (https://github.com/doconce/doconce/)
# doconce format html vmcdmc.do.txt  -->

# # Variational Monte Carlo methods

# ## Quantum Monte Carlo Motivation
# 
# We start with the variational principle.
# Given a hamiltonian $H$ and a trial wave function $\Psi_T$, the variational principle states that the expectation value of $\langle H \rangle$, defined through

# $$
# E[H]= \langle H \rangle =
#    \frac{\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R})H(\boldsymbol{R})\Psi_T(\boldsymbol{R})}
#         {\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R})\Psi_T(\boldsymbol{R})},
# $$

# is an upper bound to the ground state energy $E_0$ of the hamiltonian $H$, that is

# $$
# E_0 \le \langle H \rangle .
# $$

# In general, the integrals involved in the calculation of various  expectation values  are multi-dimensional ones. Traditional integration methods such as the Gauss-Legendre will not be adequate for say the  computation of the energy of a many-body system.
# 
# The trial wave function can be expanded in the eigenstates of the hamiltonian since they form a complete set, viz.,

# $$
# \Psi_T(\boldsymbol{R})=\sum_i a_i\Psi_i(\boldsymbol{R}),
# $$

# and assuming the set of eigenfunctions to be normalized one obtains

# $$
# \frac{\sum_{nm}a^*_ma_n \int d\boldsymbol{R}\Psi^{\ast}_m(\boldsymbol{R})H(\boldsymbol{R})\Psi_n(\boldsymbol{R})}
#         {\sum_{nm}a^*_ma_n \int d\boldsymbol{R}\Psi^{\ast}_m(\boldsymbol{R})\Psi_n(\boldsymbol{R})} =\frac{\sum_{n}a^2_n E_n}
#         {\sum_{n}a^2_n} \ge E_0,
# $$

# where we used that $H(\boldsymbol{R})\Psi_n(\boldsymbol{R})=E_n\Psi_n(\boldsymbol{R})$.
# In general, the integrals involved in the calculation of various  expectation
# values  are multi-dimensional ones. 
# The variational principle yields the lowest state of a given symmetry.
# 
# In most cases, a wave function has only small values in large parts of 
# configuration space, and a straightforward procedure which uses
# homogenously distributed random points in configuration space 
# will most likely lead to poor results. This may suggest that some kind
# of importance sampling combined with e.g., the Metropolis algorithm 
# may be  a more efficient way of obtaining the ground state energy.
# The hope is then that those regions of configurations space where
# the wave function assumes appreciable values are sampled more 
# efficiently. 
# 
# The tedious part in a VMC calculation is the search for the variational
# minimum. A good knowledge of the system is required in order to carry out
# reasonable VMC calculations. This is not always the case, 
# and often VMC calculations 
# serve rather as the starting
# point for so-called diffusion Monte Carlo calculations (DMC). DMC is a way of
# solving exactly the many-body Schroedinger equation by means of 
# a stochastic procedure. A good guess on the binding energy
# and its wave function is however necessary. 
# A carefully performed VMC calculation can aid in this context. 
# 
# The basic recipe in a VMC calculation consists of the following elements:
# 
# * Construct first a trial wave function $\psi_T(\boldsymbol{R},\boldsymbol{\alpha})$,  for a many-body system consisting of $N$ particles located at positions  $\boldsymbol{R}=(\boldsymbol{R}_1,\dots ,\boldsymbol{R}_N)$. The trial wave function depends on $\alpha$ variational parameters $\boldsymbol{\alpha}=(\alpha_1,\dots ,\alpha_M)$.
# 
# * Then we evaluate the expectation value of the hamiltonian $H$

# $$
# E[H]=\langle H \rangle =
#    \frac{\int d\boldsymbol{R}\Psi^{\ast}_{T}(\boldsymbol{R},\boldsymbol{\alpha})H(\boldsymbol{R})\Psi_{T}(\boldsymbol{R},\boldsymbol{\alpha})}
#         {\int d\boldsymbol{R}\Psi^{\ast}_{T}(\boldsymbol{R},\boldsymbol{\alpha})\Psi_{T}(\boldsymbol{R},\boldsymbol{\alpha})}.
# $$

# * Thereafter we vary $\alpha$ according to some minimization algorithm and return to the first step.
# 
# With a trial wave function $\psi_T(\boldsymbol{R})$ we can in turn construct the quantum mechanical probability distribution

# $$
# P(\boldsymbol{R})= \frac{\left|\psi_T(\boldsymbol{R})\right|^2}{\int \left|\psi_T(\boldsymbol{R})\right|^2d\boldsymbol{R}}.
# $$

# This is our new probability distribution function  (PDF).
# The approximation to the expectation value of the Hamiltonian is now

# $$
# E[H(\boldsymbol{\alpha})] = 
#    \frac{\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R},\boldsymbol{\alpha})H(\boldsymbol{R})\Psi_T(\boldsymbol{R},\boldsymbol{\alpha})}
#         {\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R},\boldsymbol{\alpha})\Psi_T(\boldsymbol{R},\boldsymbol{\alpha})}.
# $$

# Define a new quantity

# <!-- Equation labels as ordinary links -->
# <div id="eq:locale1"></div>
# 
# $$
# E_L(\boldsymbol{R},\boldsymbol{\alpha})=\frac{1}{\psi_T(\boldsymbol{R},\boldsymbol{\alpha})}H\psi_T(\boldsymbol{R},\boldsymbol{\alpha}),
# \label{eq:locale1} \tag{1}
# $$

# called the local energy, which, together with our trial PDF yields

# <!-- Equation labels as ordinary links -->
# <div id="eq:vmc1"></div>
# 
# $$
# E[H(\boldsymbol{\alpha})]=\int P(\boldsymbol{R})E_L(\boldsymbol{R}) d\boldsymbol{R}\approx \frac{1}{N}\sum_{i=1}^NP(\boldsymbol{R_i},\boldsymbol{\alpha})E_L(\boldsymbol{R_i},\boldsymbol{\alpha})
# \label{eq:vmc1} \tag{2}
# $$

# with $N$ being the number of Monte Carlo samples.
# 
# The Algorithm for performing a variational Monte Carlo calculations runs thus as this
# 
#    * Initialisation: Fix the number of Monte Carlo steps. Choose an initial $\boldsymbol{R}$ and variational parameters $\alpha$ and calculate $\left|\psi_T^{\alpha}(\boldsymbol{R})\right|^2$. 
# 
#    * Initialise the energy and the variance and start the Monte Carlo calculation.
# 
#       * Calculate  a trial position  $\boldsymbol{R}_p=\boldsymbol{R}+r*step$ where $r$ is a random variable $r \in [0,1]$.
# 
#       * Metropolis algorithm to accept or reject this move  $w = P(\boldsymbol{R}_p)/P(\boldsymbol{R})$.
# 
#       * If the step is accepted, then we set $\boldsymbol{R}=\boldsymbol{R}_p$. 
# 
#       * Update averages
# 
#    * Finish and compute final averages.
# 
# Observe that the jumping in space is governed by the variable *step*. This is Called brute-force sampling.
# Need importance sampling to get more relevant sampling, see lectures below.

# ### Quantum Monte Carlo: hydrogen atom
# 
# The radial Schroedinger equation for the hydrogen atom can be
# written as

# $$
# -\frac{\hbar^2}{2m}\frac{\partial^2 u(r)}{\partial r^2}-
# \left(\frac{ke^2}{r}-\frac{\hbar^2l(l+1)}{2mr^2}\right)u(r)=Eu(r),
# $$

# or with dimensionless variables

# <!-- Equation labels as ordinary links -->
# <div id="eq:hydrodimless1"></div>
# 
# $$
# -\frac{1}{2}\frac{\partial^2 u(\rho)}{\partial \rho^2}-
# \frac{u(\rho)}{\rho}+\frac{l(l+1)}{2\rho^2}u(\rho)-\lambda u(\rho)=0,
# \label{eq:hydrodimless1} \tag{3}
# $$

# with the hamiltonian

# $$
# H=-\frac{1}{2}\frac{\partial^2 }{\partial \rho^2}-
# \frac{1}{\rho}+\frac{l(l+1)}{2\rho^2}.
# $$

# Use variational parameter $\alpha$ in the trial
# wave function

# <!-- Equation labels as ordinary links -->
# <div id="eq:trialhydrogen"></div>
# 
# $$
# u_T^{\alpha}(\rho)=\alpha\rho e^{-\alpha\rho}. 
# \label{eq:trialhydrogen} \tag{4}
# $$

# Inserting this wave function into the expression for the
# local energy $E_L$ gives

# $$
# E_L(\rho)=-\frac{1}{\rho}-
#               \frac{\alpha}{2}\left(\alpha-\frac{2}{\rho}\right).
# $$

# A simple variational Monte Carlo calculation results in
# <table class="dotable" border="1">
# <thead>
# <tr><th align="center">  $\alpha$ </th> <th align="center">$\langle H \rangle $</th> <th align="center"> $\sigma^2$</th> <th align="center">$\sigma/\sqrt{N}$</th> </tr>
# </thead>
# <tbody>
# <tr><td align="center">   7.00000E-01    </td> <td align="center">   -4.57759E-01            </td> <td align="center">   4.51201E-02    </td> <td align="center">   6.71715E-04          </td> </tr>
# <tr><td align="center">   8.00000E-01    </td> <td align="center">   -4.81461E-01            </td> <td align="center">   3.05736E-02    </td> <td align="center">   5.52934E-04          </td> </tr>
# <tr><td align="center">   9.00000E-01    </td> <td align="center">   -4.95899E-01            </td> <td align="center">   8.20497E-03    </td> <td align="center">   2.86443E-04          </td> </tr>
# <tr><td align="center">   1.00000E-00    </td> <td align="center">   -5.00000E-01            </td> <td align="center">   0.00000E+00    </td> <td align="center">   0.00000E+00          </td> </tr>
# <tr><td align="center">   1.10000E+00    </td> <td align="center">   -4.93738E-01            </td> <td align="center">   1.16989E-02    </td> <td align="center">   3.42036E-04          </td> </tr>
# <tr><td align="center">   1.20000E+00    </td> <td align="center">   -4.75563E-01            </td> <td align="center">   8.85899E-02    </td> <td align="center">   9.41222E-04          </td> </tr>
# <tr><td align="center">   1.30000E+00    </td> <td align="center">   -4.54341E-01            </td> <td align="center">   1.45171E-01    </td> <td align="center">   1.20487E-03          </td> </tr>
# </tbody>
# </table>
# 
# We note that at $\alpha=1$ we obtain the exact
# result, and the variance is zero, as it should. The reason is that 
# we then have the exact wave function, and the action of the hamiltionan
# on the wave function

# $$
# H\psi = \mathrm{constant}\times \psi,
# $$

# yields just a constant. The integral which defines various 
# expectation values involving moments of the hamiltonian becomes then

# $$
# \langle H^n \rangle =
#    \frac{\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R})H^n(\boldsymbol{R})\Psi_T(\boldsymbol{R})}
#         {\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R})\Psi_T(\boldsymbol{R})}=
# \mathrm{constant}\times\frac{\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R})\Psi_T(\boldsymbol{R})}
#         {\int d\boldsymbol{R}\Psi^{\ast}_T(\boldsymbol{R})\Psi_T(\boldsymbol{R})}=\mathrm{constant}.
# $$

# **This gives an important information: the exact wave function leads to zero variance!**
# Variation is then performed by minimizing both the energy and the variance.
# 
# For bosons in a harmonic oscillator-like  trap we will use is a spherical (S)
#  or an elliptical (E) harmonic trap in one, two and finally three
#  dimensions, with the latter given by

# <!-- Equation labels as ordinary links -->
# <div id="trap_eqn"></div>
# 
# $$
# \begin{equation}
#  V_{ext}(\mathbf{r}) = \Bigg\{
#  \begin{array}{ll}
# 	 \frac{1}{2}m\omega_{ho}^2r^2 & (S)\\
#  \strut
# 	 \frac{1}{2}m[\omega_{ho}^2(x^2+y^2) + \omega_z^2z^2] & (E)
# \label{trap_eqn} \tag{5}
#  \end{array}
#  \end{equation}
# $$

# where (S) stands for symmetric and

# <!-- Equation labels as ordinary links -->
# <div id="_auto1"></div>
# 
# $$
# \begin{equation}
#      \hat{H} = \sum_i^N \left(
# 	 \frac{-\hbar^2}{2m}
# 	 { \bigtriangledown }_{i}^2 +
# 	 V_{ext}({\bf{r}}_i)\right)  +
# 	 \sum_{i<j}^{N} V_{int}({\bf{r}}_i,{\bf{r}}_j),
# \label{_auto1} \tag{6}
# \end{equation}
# $$

# as the two-body Hamiltonian of the system.  
# 
#  We will represent the inter-boson interaction by a pairwise, repulsive potential

# <!-- Equation labels as ordinary links -->
# <div id="_auto2"></div>
# 
# $$
# \begin{equation}
#  V_{int}(|\mathbf{r}_i-\mathbf{r}_j|) =  \Bigg\{
#  \begin{array}{ll}
# 	 \infty & {|\mathbf{r}_i-\mathbf{r}_j|} \leq {a}\\
# 	 0 & {|\mathbf{r}_i-\mathbf{r}_j|} > {a}
#  \end{array}
# \label{_auto2} \tag{7}
# \end{equation}
# $$

# where $a$ is the so-called hard-core diameter of the bosons.
#  Clearly, $V_{int}(|\mathbf{r}_i-\mathbf{r}_j|)$ is zero if the bosons are
#  separated by a distance $|\mathbf{r}_i-\mathbf{r}_j|$ greater than $a$ but
#  infinite if they attempt to come within a distance $|\mathbf{r}_i-\mathbf{r}_j| \leq a$.
# 
#  Our trial wave function for the ground state with $N$ atoms is given by

# <!-- Equation labels as ordinary links -->
# <div id="eq:trialwf"></div>
# 
# $$
# \begin{equation}
#  \Psi_T(\mathbf{R})=\Psi_T(\mathbf{r}_1, \mathbf{r}_2, \dots \mathbf{r}_N,\alpha,\beta)=\prod_i g(\alpha,\beta,\mathbf{r}_i)\prod_{i<j}f(a,|\mathbf{r}_i-\mathbf{r}_j|),
# \label{eq:trialwf} \tag{8}
#  \end{equation}
# $$

# where $\alpha$ and $\beta$ are variational parameters. The
#  single-particle wave function is proportional to the harmonic
#  oscillator function for the ground state

# <!-- Equation labels as ordinary links -->
# <div id="_auto3"></div>
# 
# $$
# \begin{equation}
#     g(\alpha,\beta,\mathbf{r}_i)= \exp{[-\alpha(x_i^2+y_i^2+\beta z_i^2)]}.
# \label{_auto3} \tag{9}
# \end{equation}
# $$

# For spherical traps we have $\beta = 1$ and for non-interacting
# bosons ($a=0$) we have $\alpha = 1/2a_{ho}^2$.  The correlation wave
#  function is

# <!-- Equation labels as ordinary links -->
# <div id="_auto4"></div>
# 
# $$
# \begin{equation}
#     f(a,|\mathbf{r}_i-\mathbf{r}_j|)=\Bigg\{
#  \begin{array}{ll}
# 	 0 & {|\mathbf{r}_i-\mathbf{r}_j|} \leq {a}\\
# 	 (1-\frac{a}{|\mathbf{r}_i-\mathbf{r}_j|}) & {|\mathbf{r}_i-\mathbf{r}_j|} > {a}.
#  \end{array}
# \label{_auto4} \tag{10}
# \end{equation}
# $$

# ### A simple Python code that solves the two-boson or two-fermion case in two-dimensions

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

#Trial wave function for quantum dots in two dims
def WaveFunction(r,alpha,beta):
    r1 = r[0,0]**2 + r[0,1]**2
    r2 = r[1,0]**2 + r[1,1]**2
    r12 = sqrt((r[0,0]-r[1,0])**2 + (r[0,1]-r[1,1])**2)
    deno = r12/(1+beta*r12)
    return exp(-0.5*alpha*(r1+r2)+deno)

#Local energy  for quantum dots in two dims, using analytical local energy
def LocalEnergy(r,alpha,beta):
    
    r1 = (r[0,0]**2 + r[0,1]**2)
    r2 = (r[1,0]**2 + r[1,1]**2)
    r12 = sqrt((r[0,0]-r[1,0])**2 + (r[0,1]-r[1,1])**2)
    deno = 1.0/(1+beta*r12)
    deno2 = deno*deno
    return 0.5*(1-alpha*alpha)*(r1 + r2) +2.0*alpha + 1.0/r12+deno2*(alpha*r12-deno2+2*beta*deno-1.0/r12)

# The Monte Carlo sampling with the Metropolis algo
def MonteCarloSampling():

    NumberMCcycles= 100000
    StepSize = 1.0
    # positions
    PositionOld = np.zeros((NumberParticles,Dimension), np.double)
    PositionNew = np.zeros((NumberParticles,Dimension), np.double)
    # seed for rng generator
    seed()
    # start variational parameter
    alpha = 0.9
    for ia in range(MaxVariations):
        alpha += .025
        AlphaValues[ia] = alpha
        beta = 0.2 
        for jb in range(MaxVariations):
            beta += .01
            BetaValues[jb] = beta
            energy = energy2 = 0.0
            DeltaE = 0.0
            #Initial position
            for i in range(NumberParticles):
                for j in range(Dimension):
                    PositionOld[i,j] = StepSize * (random() - .5)
            wfold = WaveFunction(PositionOld,alpha,beta)

            #Loop over MC MCcycles
            for MCcycle in range(NumberMCcycles):
                #Trial position
                for i in range(NumberParticles):
                    for j in range(Dimension):
                        PositionNew[i,j] = PositionOld[i,j] + StepSize * (random() - .5)
                wfnew = WaveFunction(PositionNew,alpha,beta)

                #Metropolis test to see whether we accept the move
                if random() < wfnew**2 / wfold**2:
                   PositionOld = PositionNew.copy()
                   wfold = wfnew
                   DeltaE = LocalEnergy(PositionOld,alpha,beta)
                energy += DeltaE
                energy2 += DeltaE**2

            #We calculate mean, variance and error ...
            energy /= NumberMCcycles
            energy2 /= NumberMCcycles
            variance = energy2 - energy**2
            error = sqrt(variance/NumberMCcycles)
            Energies[ia,jb] = energy    
    return Energies, AlphaValues, BetaValues


#Here starts the main program with variable declarations
NumberParticles = 2
Dimension = 2
MaxVariations = 10
Energies = np.zeros((MaxVariations,MaxVariations))
AlphaValues = np.zeros(MaxVariations)
BetaValues = np.zeros(MaxVariations)
(Energies, AlphaValues, BetaValues) = MonteCarloSampling()

# Prepare for plots
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
X, Y = np.meshgrid(AlphaValues, BetaValues)
surf = ax.plot_surface(X, Y, Energies,cmap=cm.coolwarm,linewidth=0, antialiased=False)
# Customize the z axis.
zmin = np.matrix(Energies).min()
zmax = np.matrix(Energies).max()
ax.set_zlim(zmin, zmax)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$')
ax.set_zlabel(r'$\langle E \rangle$')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# ## Quantum Monte Carlo: the helium atom
# 
# The helium atom consists of two electrons and a nucleus with
# charge $Z=2$. 
# The contribution  
# to the potential energy due to the attraction from the nucleus is

# $$
# -\frac{2ke^2}{r_1}-\frac{2ke^2}{r_2},
# $$

# and if we add the repulsion arising from the two 
# interacting electrons, we obtain the potential energy

# $$
# V(r_1, r_2)=-\frac{2ke^2}{r_1}-\frac{2ke^2}{r_2}+
#                \frac{ke^2}{r_{12}},
# $$

# with the electrons separated at a distance 
# $r_{12}=|\boldsymbol{r}_1-\boldsymbol{r}_2|$.
# 
# The hamiltonian becomes then

# $$
# \hat{H}=-\frac{\hbar^2\nabla_1^2}{2m}-\frac{\hbar^2\nabla_2^2}{2m}
#           -\frac{2ke^2}{r_1}-\frac{2ke^2}{r_2}+
#                \frac{ke^2}{r_{12}},
# $$

# and  Schroedingers equation reads

# $$
# \hat{H}\psi=E\psi.
# $$

# All observables are evaluated with respect to the probability distribution

# $$
# P(\boldsymbol{R})= \frac{\left|\psi_T(\boldsymbol{R})\right|^2}{\int \left|\psi_T(\boldsymbol{R})\right|^2d\boldsymbol{R}}.
# $$

# generated by the trial wave function.   
# The trial wave function must approximate an exact 
# eigenstate in order that accurate results are to be obtained. 
# 
# Choice of trial wave function for Helium:
# Assume $r_1 \rightarrow 0$.

# $$
# E_L(\boldsymbol{R})=\frac{1}{\psi_T(\boldsymbol{R})}H\psi_T(\boldsymbol{R})=
#      \frac{1}{\psi_T(\boldsymbol{R})}\left(-\frac{1}{2}\nabla^2_1
#      -\frac{Z}{r_1}\right)\psi_T(\boldsymbol{R}) + \mathrm{finite \hspace{0.1cm}terms}.
# $$

# $$
# E_L(R)=
#     \frac{1}{\mathbf{R}_T(r_1)}\left(-\frac{1}{2}\frac{d^2}{dr_1^2}-
#      \frac{1}{r_1}\frac{d}{dr_1}
#      -\frac{Z}{r_1}\right)\mathbf{R}_T(r_1) + \mathrm{finite\hspace{0.1cm} terms}
# $$

# For small values of $r_1$, the terms which dominate are

# $$
# \lim_{r_1 \rightarrow 0}E_L(R)=
#     \frac{1}{\mathbf{R}_T(r_1)}\left(-
#      \frac{1}{r_1}\frac{d}{dr_1}
#      -\frac{Z}{r_1}\right)\mathbf{R}_T(r_1),
# $$

# since the second derivative does not diverge due to the finiteness of  $\Psi$ at the origin.
# 
# This results in

# $$
# \frac{1}{\mathbf{R}_T(r_1)}\frac{d \mathbf{R}_T(r_1)}{dr_1}=-Z,
# $$

# and

# $$
# \mathbf{R}_T(r_1)\propto e^{-Zr_1}.
# $$

# A similar condition applies to electron 2 as well. 
# For orbital momenta $l > 0$ we have

# $$
# \frac{1}{\mathbf{R}_T(r)}\frac{d \mathbf{R}_T(r)}{dr}=-\frac{Z}{l+1}.
# $$

# Similarly, studying the case $r_{12}\rightarrow 0$ we can write 
# a possible trial wave function as

# <!-- Equation labels as ordinary links -->
# <div id="eq:wavehelium2"></div>
# 
# $$
# \psi_T(\boldsymbol{R})=e^{-\alpha(r_1+r_2)}e^{\beta r_{12}}.
# \label{eq:wavehelium2} \tag{11}
# $$

# The last equation can be generalized to

# $$
# \psi_T(\boldsymbol{R})=\phi(\boldsymbol{r}_1)\phi(\boldsymbol{r}_2)\dots\phi(\boldsymbol{r}_N)
#                    \prod_{i < j}f(r_{ij}),
# $$

# for a system with $N$ electrons or particles. 
# 
# During the development of our code we need to make several checks. It is also very instructive to compute a closed form expression for the local energy. Since our wave function is rather simple  it is straightforward
# to find an analytic expressions.  Consider first the case of the simple helium function

# $$
# \Psi_T(\boldsymbol{r}_1,\boldsymbol{r}_2) = e^{-\alpha(r_1+r_2)}
# $$

# The local energy is for this case

# $$
# E_{L1} = \left(\alpha-Z\right)\left(\frac{1}{r_1}+\frac{1}{r_2}\right)+\frac{1}{r_{12}}-\alpha^2
# $$

# which gives an expectation value for the local energy given by

# $$
# \langle E_{L1} \rangle = \alpha^2-2\alpha\left(Z-\frac{5}{16}\right)
# $$

# With closed form formulae we  can speed up the computation of the correlation. In our case
# we write it as

# $$
# \Psi_C= \exp{\left\{\sum_{i < j}\frac{ar_{ij}}{1+\beta r_{ij}}\right\}},
# $$

# which means that the gradient needed for the so-called quantum force and local energy 
# can be calculated analytically.
# This will speed up your code since the computation of the correlation part and the Slater determinant are the most 
# time consuming parts in your code.  
# 
# We will refer to this correlation function as $\Psi_C$ or the *linear Pade-Jastrow*.
# 
# We can test this by computing the local energy for our helium wave function

# $$
# \psi_{T}(\boldsymbol{r}_1,\boldsymbol{r}_2) = 
#    \exp{\left(-\alpha(r_1+r_2)\right)}
#    \exp{\left(\frac{r_{12}}{2(1+\beta r_{12})}\right)},
# $$

# with $\alpha$ and $\beta$ as variational parameters.
# 
# The local energy is for this case

# $$
# E_{L2} = E_{L1}+\frac{1}{2(1+\beta r_{12})^2}\left\{\frac{\alpha(r_1+r_2)}{r_{12}}(1-\frac{\boldsymbol{r}_1\boldsymbol{r}_2}{r_1r_2})-\frac{1}{2(1+\beta r_{12})^2}-\frac{2}{r_{12}}+\frac{2\beta}{1+\beta r_{12}}\right\}
# $$

# It is very useful to test your code against these expressions. It means also that you don't need to
# compute a derivative numerically as discussed in the code example below. 
# 
# For the computation of various derivatives with different types of wave functions, you will find it useful to use python with symbolic python, that is sympy, see [online manual](http://docs.sympy.org/latest/index.html).  Using sympy allows you autogenerate both Latex code as well c++, python or Fortran codes. Here you will find some simple examples. We choose 
# the $2s$ hydrogen-orbital  (not normalized) as an example

# $$
# \phi_{2s}(\boldsymbol{r}) = (Zr - 2)\exp{-(\frac{1}{2}Zr)},
# $$

# with $ r^2 = x^2 + y^2 + z^2$.

# In[2]:


from sympy import symbols, diff, exp, sqrt
x, y, z, Z = symbols('x y z Z')
r = sqrt(x*x + y*y + z*z)
r
phi = (Z*r - 2)*exp(-Z*r/2)
phi
diff(phi, x)


# This doesn't look very nice, but sympy provides several functions that allow for improving and simplifying the output.
# 
# We can improve our output by factorizing and substituting expressions

# In[3]:


from sympy import symbols, diff, exp, sqrt, factor, Symbol, printing
x, y, z, Z = symbols('x y z Z')
r = sqrt(x*x + y*y + z*z)
phi = (Z*r - 2)*exp(-Z*r/2)
R = Symbol('r') #Creates a symbolic equivalent of r
#print latex and c++ code
print printing.latex(diff(phi, x).factor().subs(r, R))
print printing.ccode(diff(phi, x).factor().subs(r, R))


# We can in turn look at second derivatives

# In[4]:


from sympy import symbols, diff, exp, sqrt, factor, Symbol, printing
x, y, z, Z = symbols('x y z Z')
r = sqrt(x*x + y*y + z*z)
phi = (Z*r - 2)*exp(-Z*r/2)
R = Symbol('r') #Creates a symbolic equivalent of r
(diff(diff(phi, x), x) + diff(diff(phi, y), y) + diff(diff(phi, z), z)).factor().subs(r, R)
# Collect the Z values
(diff(diff(phi, x), x) + diff(diff(phi, y), y) +diff(diff(phi, z), z)).factor().collect(Z).subs(r, R)
# Factorize also the r**2 terms
(diff(diff(phi, x), x) + diff(diff(phi, y), y) + diff(diff(phi, z), z)).factor().collect(Z).subs(r, R).subs(r**2, R**2).factor()
print printing.ccode((diff(diff(phi, x), x) + diff(diff(phi, y), y) + diff(diff(phi, z), z)).factor().collect(Z).subs(r, R).subs(r**2, R**2).factor())


# With some practice this allows one to be able to check one's own calculation and translate automatically into code lines.

# ## The Metropolis algorithm
# 
# The Metropolis algorithm , see [the original article](http://scitation.aip.org/content/aip/journal/jcp/21/6/10.1063/1.1699114) was invented by Metropolis et. al
# and is often simply called the Metropolis algorithm.
# It is a method to sample a normalized probability
# distribution by a stochastic process. We define $\mathbf{P}_i^{(n)}$ to
# be the probability for finding the system in the state $i$ at step $n$.
# The algorithm is then
# 
# * Sample a possible new state $j$ with some probability $T_{i\rightarrow j}$.
# 
# * Accept the new state $j$ with probability $A_{i \rightarrow j}$ and use it as the next sample. With probability $1-A_{i\rightarrow j}$ the move is rejected and the original state $i$ is used again as a sample.
# 
# We wish to derive the required properties of $T$ and $A$ such that
# $\mathbf{P}_i^{(n\rightarrow \infty)} \rightarrow p_i$ so that starting
# from any distribution, the method converges to the correct distribution.
# Note that the description here is for a discrete probability distribution.
# Replacing probabilities $p_i$ with expressions like $p(x_i)dx_i$ will
# take all of these over to the corresponding continuum expressions.
# 
# The dynamical equation for $\mathbf{P}_i^{(n)}$ can be written directly from
# the description above. The probability of being in the state $i$ at step $n$
# is given by the probability of being in any state $j$ at the previous step,
# and making an accepted transition to $i$ added to the probability of
# being in the state $i$, making a transition to any state $j$ and
# rejecting the move:

# $$
# \mathbf{P}^{(n)}_i = \sum_j \left [
# \mathbf{P}^{(n-1)}_jT_{j\rightarrow i} A_{j\rightarrow i} 
# +\mathbf{P}^{(n-1)}_iT_{i\rightarrow j}\left ( 1- A_{i\rightarrow j} \right)
# \right ] \,.
# $$

# Since the probability of making some transition must be 1,
# $\sum_j T_{i\rightarrow j} = 1$, and the above equation becomes

# $$
# \mathbf{P}^{(n)}_i = \mathbf{P}^{(n-1)}_i +
#  \sum_j \left [
# \mathbf{P}^{(n-1)}_jT_{j\rightarrow i} A_{j\rightarrow i} 
# -\mathbf{P}^{(n-1)}_iT_{i\rightarrow j}A_{i\rightarrow j}
# \right ] \,.
# $$

# For large $n$ we require that $\mathbf{P}^{(n\rightarrow \infty)}_i = p_i$,
# the desired probability distribution. Taking this limit, gives the
# balance requirement

# $$
# \sum_j \left [
# p_jT_{j\rightarrow i} A_{j\rightarrow i}
# -p_iT_{i\rightarrow j}A_{i\rightarrow j}
# \right ] = 0 \,.
# $$

# The balance requirement is very weak. Typically the much stronger detailed
# balance requirement is enforced, that is rather than the sum being
# set to zero, we set each term separately to zero and use this
# to determine the acceptance probabilities. Rearranging, the result is

# $$
# \frac{ A_{j\rightarrow i}}{A_{i\rightarrow j}}
# = \frac{p_iT_{i\rightarrow j}}{ p_jT_{j\rightarrow i}} \,.
# $$

# The Metropolis choice is to maximize the $A$ values, that is

# $$
# A_{j \rightarrow i} = \min \left ( 1,
# \frac{p_iT_{i\rightarrow j}}{ p_jT_{j\rightarrow i}}\right ).
# $$

# Other choices are possible, but they all correspond to multilplying
# $A_{i\rightarrow j}$ and $A_{j\rightarrow i}$ by the same constant
# smaller than unity.\footnote{The penalty function method uses just such
# a factor to compensate for $p_i$ that are evaluated stochastically
# and are therefore noisy.}
# 
# Having chosen the acceptance probabilities, we have guaranteed that
# if the  $\mathbf{P}_i^{(n)}$ has equilibrated, that is if it is equal to $p_i$,
# it will remain equilibrated. Next we need to find the circumstances for
# convergence to equilibrium.
# 
# The dynamical equation can be written as

# $$
# \mathbf{P}^{(n)}_i = \sum_j M_{ij}\mathbf{P}^{(n-1)}_j
# $$

# with the matrix $M$ given by

# $$
# M_{ij} = \delta_{ij}\left [ 1 -\sum_k T_{i\rightarrow k} A_{i \rightarrow k}
# \right ] + T_{j\rightarrow i} A_{j\rightarrow i} \,.
# $$

# Summing over $i$ shows that $\sum_i M_{ij} = 1$, and since
# $\sum_k T_{i\rightarrow k} = 1$, and $A_{i \rightarrow k} \leq 1$, the
# elements of the matrix satisfy $M_{ij} \geq 0$. The matrix $M$ is therefore
# a stochastic matrix.
# 
# The Metropolis method is simply the power method for computing the
# right eigenvector of $M$ with the largest magnitude eigenvalue.
# By construction, the correct probability distribution is a right eigenvector
# with eigenvalue 1. Therefore, for the Metropolis method to converge
# to this result, we must show that $M$ has only one eigenvalue with this
# magnitude, and all other eigenvalues are smaller.

# ## Importance sampling
# 
# We need to replace the brute force
# Metropolis algorithm with a walk in coordinate space biased by the trial wave function.
# This approach is based on the Fokker-Planck equation and the Langevin equation for generating a trajectory in coordinate space.  The link between the Fokker-Planck equation and the Langevin equations are explained, only partly, in the slides below.
# An excellent reference on topics like Brownian motion, Markov chains, the Fokker-Planck equation and the Langevin equation is the text by  [Van Kampen](http://www.elsevier.com/books/stochastic-processes-in-physics-and-chemistry/van-kampen/978-0-444-52965-7)
# Here we will focus first on the implementation part first.
# 
# For a diffusion process characterized by a time-dependent probability density $P(x,t)$ in one dimension the Fokker-Planck
# equation reads (for one particle /walker)

# $$
# \frac{\partial P}{\partial t} = D\frac{\partial }{\partial x}\left(\frac{\partial }{\partial x} -F\right)P(x,t),
# $$

# where $F$ is a drift term and $D$ is the diffusion coefficient. 
# 
# The new positions in coordinate space are given as the solutions of the Langevin equation using Euler's method, namely,
# we go from the Langevin equation

# $$
# \frac{\partial x(t)}{\partial t} = DF(x(t)) +\eta,
# $$

# with $\eta$ a random variable,
# yielding a new position

# $$
# y = x+DF(x)\Delta t +\xi\sqrt{\Delta t},
# $$

# where $\xi$ is gaussian random variable and $\Delta t$ is a chosen time step. 
# The quantity $D$ is, in atomic units, equal to $1/2$ and comes from the factor $1/2$ in the kinetic energy operator. Note that $\Delta t$ is to be viewed as a parameter. Values of $\Delta t \in [0.001,0.01]$ yield in general rather stable values of the ground state energy.  
# 
# The process of isotropic diffusion characterized by a time-dependent probability density $P(\mathbf{x},t)$ obeys (as an approximation) the so-called Fokker-Planck equation

# $$
# \frac{\partial P}{\partial t} = \sum_i D\frac{\partial }{\partial \mathbf{x_i}}\left(\frac{\partial }{\partial \mathbf{x_i}} -\mathbf{F_i}\right)P(\mathbf{x},t),
# $$

# where $\mathbf{F_i}$ is the $i^{th}$ component of the drift term (drift velocity) caused by an external potential, and $D$ is the diffusion coefficient. The convergence to a stationary probability density can be obtained by setting the left hand side to zero. The resulting equation will be satisfied if and only if all the terms of the sum are equal zero,

# $$
# \frac{\partial^2 P}{\partial {\mathbf{x_i}^2}} = P\frac{\partial}{\partial {\mathbf{x_i}}}\mathbf{F_i} + \mathbf{F_i}\frac{\partial}{\partial {\mathbf{x_i}}}P.
# $$

# The drift vector should be of the form $\mathbf{F} = g(\mathbf{x}) \frac{\partial P}{\partial \mathbf{x}}$. Then,

# $$
# \frac{\partial^2 P}{\partial {\mathbf{x_i}^2}} = P\frac{\partial g}{\partial P}\left( \frac{\partial P}{\partial {\mathbf{x}_i}}  \right)^2 + P g \frac{\partial ^2 P}{\partial {\mathbf{x}_i^2}}  + g \left( \frac{\partial P}{\partial {\mathbf{x}_i}}  \right)^2.
# $$

# The condition of stationary density means that the left hand side equals zero. In other words, the terms containing first and second derivatives have to cancel each other. It is possible only if $g = \frac{1}{P}$, which yields

# $$
# \mathbf{F} = 2\frac{1}{\Psi_T}\nabla\Psi_T,
# $$

# which is known as the so-called *quantum force*. This term is responsible for pushing the walker towards regions of configuration space where the trial wave function is large, increasing the efficiency of the simulation in contrast to the Metropolis algorithm where the walker has the same probability of moving in every direction.
# 
# The Fokker-Planck equation yields a (the solution to the equation) transition probability given by the Green's function

# $$
# G(y,x,\Delta t) = \frac{1}{(4\pi D\Delta t)^{3N/2}} \exp{\left(-(y-x-D\Delta t F(x))^2/4D\Delta t\right)}
# $$

# which in turn means that our brute force Metropolis algorithm

# $$
# A(y,x) = \mathrm{min}(1,q(y,x))),
# $$

# with $q(y,x) = |\Psi_T(y)|^2/|\Psi_T(x)|^2$ is now replaced by the [Metropolis-Hastings algorithm](http://scitation.aip.org/content/aip/journal/jcp/21/6/10.1063/1.1699114) as well as [Hasting's article](http://biomet.oxfordjournals.org/content/57/1/97.abstract),

# $$
# q(y,x) = \frac{G(x,y,\Delta t)|\Psi_T(y)|^2}{G(y,x,\Delta t)|\Psi_T(x)|^2}
# $$

# ## Importance sampling, program elements
# 
# The general derivative formula of the Jastrow factor is (the subscript $C$ stands for Correlation)

# $$
# \frac{1}{\Psi_C}\frac{\partial \Psi_C}{\partial x_k} =
# \sum_{i=1}^{k-1}\frac{\partial g_{ik}}{\partial x_k}
# +
# \sum_{i=k+1}^{N}\frac{\partial g_{ki}}{\partial x_k}
# $$

# However, 
# with our written in way which can be reused later as

# $$
# \Psi_C=\prod_{i< j}g(r_{ij})= \exp{\left\{\sum_{i<j}f(r_{ij})\right\}},
# $$

# the gradient needed for the quantum force and local energy is easy to compute.  
# The function $f(r_{ij})$ will depends on the system under study. In the equations below we will keep this general form.
# 
# In the Metropolis/Hasting algorithm, the *acceptance ratio* determines the probability for a particle  to be accepted at a new position. The ratio of the trial wave functions evaluated at the new and current positions is given by ($OB$ for the onebody  part)

# $$
# R \equiv \frac{\Psi_{T}^{new}}{\Psi_{T}^{old}} = 
# \frac{\Psi_{OB}^{new}}{\Psi_{OB}^{old}}\frac{\Psi_{C}^{new}}{\Psi_{C}^{old}}
# $$

# Here $\Psi_{OB}$ is our onebody part (Slater determinant or product of boson single-particle states)  while $\Psi_{C}$ is our correlation function, or Jastrow factor. 
# We need to optimize the $\nabla \Psi_T / \Psi_T$ ratio and the second derivative as well, that is
# the $\mathbf{\nabla}^2 \Psi_T/\Psi_T$ ratio. The first is needed when we compute the so-called quantum force in importance sampling.
# The second is needed when we compute the kinetic energy term of the local energy.

# $$
# \frac{\mathbf{\mathbf{\nabla}}  \Psi}{\Psi}  = \frac{\mathbf{\nabla}  (\Psi_{OB} \, \Psi_{C})}{\Psi_{OB} \, \Psi_{C}}  =  \frac{ \Psi_C \mathbf{\nabla}  \Psi_{OB} + \Psi_{OB} \mathbf{\nabla}  \Psi_{C}}{\Psi_{OB} \Psi_{C}} = \frac{\mathbf{\nabla}  \Psi_{OB}}{\Psi_{OB}} + \frac{\mathbf{\nabla}   \Psi_C}{ \Psi_C}
# $$

# The expectation value of the kinetic energy expressed in atomic units for electron $i$ is

# $$
# \langle \hat{K}_i \rangle = -\frac{1}{2}\frac{\langle\Psi|\mathbf{\nabla}_{i}^2|\Psi \rangle}{\langle\Psi|\Psi \rangle},
# $$

# $$
# \hat{K}_i = -\frac{1}{2}\frac{\mathbf{\nabla}_{i}^{2} \Psi}{\Psi}.
# $$

# The second derivative which enters the definition of the local energy is

# $$
# \frac{\mathbf{\nabla}^2 \Psi}{\Psi}=\frac{\mathbf{\nabla}^2 \Psi_{OB}}{\Psi_{OB}} + \frac{\mathbf{\nabla}^2  \Psi_C}{ \Psi_C} + 2 \frac{\mathbf{\nabla}  \Psi_{OB}}{\Psi_{OB}}\cdot\frac{\mathbf{\nabla}   \Psi_C}{ \Psi_C}
# $$

# We discuss here how to calculate these quantities in an optimal way,
# 
# We have defined the correlated function as

# $$
# \Psi_C=\prod_{i< j}g(r_{ij})=\prod_{i< j}^Ng(r_{ij})= \prod_{i=1}^N\prod_{j=i+1}^Ng(r_{ij}),
# $$

# with 
# $r_{ij}=|\mathbf{r}_i-\mathbf{r}_j|=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2+(z_i-z_j)^2}$ in three dimensions or
# $r_{ij}=|\mathbf{r}_i-\mathbf{r}_j|=\sqrt{(x_i-x_j)^2+(y_i-y_j)^2}$ if we work with two-dimensional systems.
# 
# In our particular case we have

# $$
# \Psi_C=\prod_{i< j}g(r_{ij})=\exp{\left\{\sum_{i<j}f(r_{ij})\right\}}.
# $$

# The total number of different relative distances $r_{ij}$ is $N(N-1)/2$. In a matrix storage format, the relative distances  form a strictly upper triangular matrix

# $$
# \mathbf{r} \equiv \begin{pmatrix}
#   0 & r_{1,2} & r_{1,3} & \cdots & r_{1,N} \\
#   \vdots & 0       & r_{2,3} & \cdots & r_{2,N} \\
#   \vdots & \vdots  & 0  & \ddots & \vdots  \\
#   \vdots & \vdots  & \vdots  & \ddots  & r_{N-1,N} \\
#   0 & 0  & 0  & \cdots  & 0
#  \end{pmatrix}.
# $$

# This applies to  $\mathbf{g} = \mathbf{g}(r_{ij})$ as well. 
# 
# In our algorithm we will move one particle  at the time, say the $kth$-particle.  This sampling will be seen to be particularly efficient when we are going to compute a Slater determinant. 
# 
# We have that the ratio between Jastrow factors $R_C$ is given by

# $$
# R_{C} = \frac{\Psi_{C}^\mathrm{new}}{\Psi_{C}^\mathrm{cur}} =
# \prod_{i=1}^{k-1}\frac{g_{ik}^\mathrm{new}}{g_{ik}^\mathrm{cur}}
# \prod_{i=k+1}^{N}\frac{ g_{ki}^\mathrm{new}} {g_{ki}^\mathrm{cur}}.
# $$

# For the Pade-Jastrow form

# $$
# R_{C} = \frac{\Psi_{C}^\mathrm{new}}{\Psi_{C}^\mathrm{cur}} = 
# \frac{\exp{U_{new}}}{\exp{U_{cur}}} = \exp{\Delta U},
# $$

# where

# $$
# \Delta U =
# \sum_{i=1}^{k-1}\big(f_{ik}^\mathrm{new}-f_{ik}^\mathrm{cur}\big)
# +
# \sum_{i=k+1}^{N}\big(f_{ki}^\mathrm{new}-f_{ki}^\mathrm{cur}\big)
# $$

# One needs to develop a special algorithm 
# that runs only through the elements of the upper triangular
# matrix $\mathbf{g}$ and have $k$ as an index. 
# 
# The expression to be derived in the following is of interest when computing the quantum force and the kinetic energy. It has the form

# $$
# \frac{\mathbf{\nabla}_i\Psi_C}{\Psi_C} = \frac{1}{\Psi_C}\frac{\partial \Psi_C}{\partial x_i},
# $$

# for all dimensions and with $i$ running over all particles.
# 
# For the first derivative only $N-1$ terms survive the ratio because the $g$-terms that are not differentiated cancel with their corresponding ones in the denominator. Then,

# $$
# \frac{1}{\Psi_C}\frac{\partial \Psi_C}{\partial x_k} =
# \sum_{i=1}^{k-1}\frac{1}{g_{ik}}\frac{\partial g_{ik}}{\partial x_k}
# +
# \sum_{i=k+1}^{N}\frac{1}{g_{ki}}\frac{\partial g_{ki}}{\partial x_k}.
# $$

# An equivalent equation is obtained for the exponential form after replacing $g_{ij}$ by $\exp(f_{ij})$, yielding:

# $$
# \frac{1}{\Psi_C}\frac{\partial \Psi_C}{\partial x_k} =
# \sum_{i=1}^{k-1}\frac{\partial g_{ik}}{\partial x_k}
# +
# \sum_{i=k+1}^{N}\frac{\partial g_{ki}}{\partial x_k},
# $$

# with both expressions scaling as $\mathcal{O}(N)$.
# 
# Using the identity

# $$
# \frac{\partial}{\partial x_i}g_{ij} = -\frac{\partial}{\partial x_j}g_{ij},
# $$

# we get expressions where all the derivatives acting on the particle  are represented by the *second* index of $g$:

# $$
# \frac{1}{\Psi_C}\frac{\partial \Psi_C}{\partial x_k} =
# \sum_{i=1}^{k-1}\frac{1}{g_{ik}}\frac{\partial g_{ik}}{\partial x_k}
# -\sum_{i=k+1}^{N}\frac{1}{g_{ki}}\frac{\partial g_{ki}}{\partial x_i},
# $$

# and for the exponential case:

# $$
# \frac{1}{\Psi_C}\frac{\partial \Psi_C}{\partial x_k} =
# \sum_{i=1}^{k-1}\frac{\partial g_{ik}}{\partial x_k}
# -\sum_{i=k+1}^{N}\frac{\partial g_{ki}}{\partial x_i}.
# $$

# For correlation forms depending only on the scalar distances $r_{ij}$ we can use the chain rule. Noting that

# $$
# \frac{\partial g_{ij}}{\partial x_j} = \frac{\partial g_{ij}}{\partial r_{ij}} \frac{\partial r_{ij}}{\partial x_j} = \frac{x_j - x_i}{r_{ij}} \frac{\partial g_{ij}}{\partial r_{ij}},
# $$

# we arrive at

# $$
# \frac{1}{\Psi_C}\frac{\partial \Psi_C}{\partial x_k} = 
# \sum_{i=1}^{k-1}\frac{1}{g_{ik}} \frac{\mathbf{r_{ik}}}{r_{ik}} \frac{\partial g_{ik}}{\partial r_{ik}}
# -\sum_{i=k+1}^{N}\frac{1}{g_{ki}}\frac{\mathbf{r_{ki}}}{r_{ki}}\frac{\partial g_{ki}}{\partial r_{ki}}.
# $$

# Note that for the Pade-Jastrow form we can set $g_{ij} \equiv g(r_{ij}) = e^{f(r_{ij})} = e^{f_{ij}}$ and

# $$
# \frac{\partial g_{ij}}{\partial r_{ij}} = g_{ij} \frac{\partial f_{ij}}{\partial r_{ij}}.
# $$

# Therefore,

# $$
# \frac{1}{\Psi_{C}}\frac{\partial \Psi_{C}}{\partial x_k} =
# \sum_{i=1}^{k-1}\frac{\mathbf{r_{ik}}}{r_{ik}}\frac{\partial f_{ik}}{\partial r_{ik}}
# -\sum_{i=k+1}^{N}\frac{\mathbf{r_{ki}}}{r_{ki}}\frac{\partial f_{ki}}{\partial r_{ki}},
# $$

# where

# $$
# \mathbf{r}_{ij} = |\mathbf{r}_j - \mathbf{r}_i| = (x_j - x_i)\mathbf{e}_1 + (y_j - y_i)\mathbf{e}_2 + (z_j - z_i)\mathbf{e}_3
# $$

# is the relative distance. 
# 
# The second derivative of the Jastrow factor divided by the Jastrow factor (the way it enters the kinetic energy) is

# $$
# \left[\frac{\mathbf{\nabla}^2 \Psi_C}{\Psi_C}\right]_x =\  
# 2\sum_{k=1}^{N}
# \sum_{i=1}^{k-1}\frac{\partial^2 g_{ik}}{\partial x_k^2}\ +\ 
# \sum_{k=1}^N
# \left(
# \sum_{i=1}^{k-1}\frac{\partial g_{ik}}{\partial x_k} -
# \sum_{i=k+1}^{N}\frac{\partial g_{ki}}{\partial x_i}
# \right)^2
# $$

# But we have a simple form for the function, namely

# $$
# \Psi_{C}=\prod_{i< j}\exp{f(r_{ij})},
# $$

# and it is easy to see that for particle  $k$
# we have

# $$
# \frac{\mathbf{\nabla}^2_k \Psi_C}{\Psi_C }=
# \sum_{ij\ne k}\frac{(\mathbf{r}_k-\mathbf{r}_i)(\mathbf{r}_k-\mathbf{r}_j)}{r_{ki}r_{kj}}f'(r_{ki})f'(r_{kj})+
# \sum_{j\ne k}\left( f''(r_{kj})+\frac{2}{r_{kj}}f'(r_{kj})\right)
# $$

# ## Importance sampling, Fokker-Planck and Langevin equations
# 
# A stochastic process is simply a function of two variables, one is the time,
# the other is a stochastic variable $X$, defined by specifying
# * the set $\left\{x\right\}$ of possible values for $X$;
# 
# * the probability distribution, $w_X(x)$,  over this set, or briefly $w(x)$
# 
# The set of values $\left\{x\right\}$ for $X$ 
# may be discrete, or continuous. If the set of
# values is continuous, then $w_X (x)$ is a probability density so that 
# $w_X (x)dx$
# is the probability that one finds the stochastic variable $X$ to have values
# in the range $[x, x + dx]$ .
# 
#      An arbitrary number of other stochastic variables may be derived from
# $X$. For example, any $Y$ given by a mapping of $X$, is also a stochastic
# variable. The mapping may also be time-dependent, that is, the mapping
# depends on an additional variable $t$

# $$
# Y_X (t) = f (X, t) .
# $$

# The quantity $Y_X (t)$ is called a random function, or, since $t$ often is time,
# a stochastic process. A stochastic process is a function of two variables,
# one is the time, the other is a stochastic variable $X$. Let $x$ be one of the
# possible values of $X$ then

# $$
# y(t) = f (x, t),
# $$

# is a function of $t$, called a sample function or realization of the process.
# In physics one considers the stochastic process to be an ensemble of such
# sample functions.
# 
#      For many physical systems initial distributions of a stochastic 
# variable $y$ tend to equilibrium distributions: $w(y, t)\rightarrow w_0(y)$ 
# as $t\rightarrow\infty$. In
# equilibrium detailed balance constrains the transition rates

# $$
# W(y\rightarrow y')w(y ) = W(y'\rightarrow y)w_0 (y),
# $$

# where $W(y'\rightarrow y)$ 
# is the probability, per unit time, that the system changes
# from a state $|y\rangle$ , characterized by the value $y$ 
# for the stochastic variable $Y$ , to a state $|y'\rangle$.
# 
# Note that for a system in equilibrium the transition rate 
# $W(y'\rightarrow y)$ and
# the reverse $W(y\rightarrow y')$ may be very different. 
# 
# Consider, for instance, a simple
# system that has only two energy levels $\epsilon_0 = 0$ and 
# $\epsilon_1 = \Delta E$. 
# 
# For a system governed by the Boltzmann distribution we find (the partition function has been taken out)

# $$
# W(0\rightarrow 1)\exp{-(\epsilon_0/kT)} = W(1\rightarrow 0)\exp{-(\epsilon_1/kT)}
# $$

# We get then

# $$
# \frac{W(1\rightarrow 0)}{W(0 \rightarrow 1)}=\exp{-(\Delta E/kT)},
# $$

# which goes to zero when $T$ tends to zero.
# 
# If we assume a discrete set of events,
# our initial probability
# distribution function can be  given by

# $$
# w_i(0) = \delta_{i,0},
# $$

# and its time-development after a given time step $\Delta t=\epsilon$ is

# $$
# w_i(t) = \sum_{j}W(j\rightarrow i)w_j(t=0).
# $$

# The continuous analog to $w_i(0)$ is

# $$
# w(\mathbf{x})\rightarrow \delta(\mathbf{x}),
# $$

# where we now have generalized the one-dimensional position $x$ to a generic-dimensional  
# vector $\mathbf{x}$. The Kroenecker $\delta$ function is replaced by the $\delta$ distribution
# function $\delta(\mathbf{x})$ at  $t=0$.  
# 
# The transition from a state $j$ to a state $i$ is now replaced by a transition
# to a state with position $\mathbf{y}$ from a state with position $\mathbf{x}$. 
# The discrete sum of transition probabilities can then be replaced by an integral
# and we obtain the new distribution at a time $t+\Delta t$ as

# $$
# w(\mathbf{y},t+\Delta t)= \int W(\mathbf{y},t+\Delta t| \mathbf{x},t)w(\mathbf{x},t)d\mathbf{x},
# $$

# and after $m$ time steps we have

# $$
# w(\mathbf{y},t+m\Delta t)= \int W(\mathbf{y},t+m\Delta t| \mathbf{x},t)w(\mathbf{x},t)d\mathbf{x}.
# $$

# When equilibrium is reached we have

# $$
# w(\mathbf{y})= \int W(\mathbf{y}|\mathbf{x}, t)w(\mathbf{x})d\mathbf{x},
# $$

# that is no time-dependence. Note our change of notation for $W$
# 
# We can solve the equation for $w(\mathbf{y},t)$ by making a Fourier transform to
# momentum space. 
# The PDF $w(\mathbf{x},t)$ is related to its Fourier transform
# $\tilde{w}(\mathbf{k},t)$ through

# $$
# w(\mathbf{x},t) = \int_{-\infty}^{\infty}d\mathbf{k} \exp{(i\mathbf{kx})}\tilde{w}(\mathbf{k},t),
# $$

# and using the definition of the 
# $\delta$-function

# $$
# \delta(\mathbf{x}) = \frac{1}{2\pi} \int_{-\infty}^{\infty}d\mathbf{k} \exp{(i\mathbf{kx})},
# $$

# we see that

# $$
# \tilde{w}(\mathbf{k},0)=1/2\pi.
# $$

# We can then use the Fourier-transformed diffusion equation

# $$
# \frac{\partial \tilde{w}(\mathbf{k},t)}{\partial t} = -D\mathbf{k}^2\tilde{w}(\mathbf{k},t),
# $$

# with the obvious solution

# $$
# \tilde{w}(\mathbf{k},t)=\tilde{w}(\mathbf{k},0)\exp{\left[-(D\mathbf{k}^2t)\right)}=
#     \frac{1}{2\pi}\exp{\left[-(D\mathbf{k}^2t)\right]}.
# $$

# With the Fourier transform we obtain

# $$
# w(\mathbf{x},t)=\int_{-\infty}^{\infty}d\mathbf{k} \exp{\left[i\mathbf{kx}\right]}\frac{1}{2\pi}\exp{\left[-(D\mathbf{k}^2t)\right]}=
#     \frac{1}{\sqrt{4\pi Dt}}\exp{\left[-(\mathbf{x}^2/4Dt)\right]},
# $$

# with the normalization condition

# $$
# \int_{-\infty}^{\infty}w(\mathbf{x},t)d\mathbf{x}=1.
# $$

# The solution represents the probability of finding
# our random walker at position $\mathbf{x}$ at time $t$ if the initial distribution 
# was placed at $\mathbf{x}=0$ at $t=0$. 
# 
# There is another interesting feature worth observing. The discrete transition probability $W$
# itself is given by a binomial distribution.
# The results from the central limit theorem state that 
# transition probability in the limit $n\rightarrow \infty$ converges to the normal 
# distribution. It is then possible to show that

# $$
# W(il-jl,n\epsilon)\rightarrow W(\mathbf{y},t+\Delta t|\mathbf{x},t)=
#     \frac{1}{\sqrt{4\pi D\Delta t}}\exp{\left[-((\mathbf{y}-\mathbf{x})^2/4D\Delta t)\right]},
# $$

# and that it satisfies the normalization condition and is itself a solution
# to the diffusion equation.
# 
# Let us now assume that we have three PDFs for times $t_0 < t' < t$, that is
# $w(\mathbf{x}_0,t_0)$, $w(\mathbf{x}',t')$ and $w(\mathbf{x},t)$.
# We have then

# $$
# w(\mathbf{x},t)= \int_{-\infty}^{\infty} W(\mathbf{x}.t|\mathbf{x}'.t')w(\mathbf{x}',t')d\mathbf{x}',
# $$

# and

# $$
# w(\mathbf{x},t)= \int_{-\infty}^{\infty} W(\mathbf{x}.t|\mathbf{x}_0.t_0)w(\mathbf{x}_0,t_0)d\mathbf{x}_0,
# $$

# and

# $$
# w(\mathbf{x}',t')= \int_{-\infty}^{\infty} W(\mathbf{x}'.t'|\mathbf{x}_0,t_0)w(\mathbf{x}_0,t_0)d\mathbf{x}_0.
# $$

# We can combine these equations and arrive at the famous Einstein-Smoluchenski-Kolmogorov-Chapman (ESKC) relation

# $$
# W(\mathbf{x}t|\mathbf{x}_0t_0)  = \int_{-\infty}^{\infty} W(\mathbf{x},t|\mathbf{x}',t')W(\mathbf{x}',t'|\mathbf{x}_0,t_0)d\mathbf{x}'.
# $$

# We can replace the spatial dependence with a dependence upon say the velocity
# (or momentum), that is we have

# $$
# W(\mathbf{v},t|\mathbf{v}_0,t_0)  = \int_{-\infty}^{\infty} W(\mathbf{v},t|\mathbf{v}',t')W(\mathbf{v}',t'|\mathbf{v}_0,t_0)d\mathbf{x}'.
# $$

# We will now derive the Fokker-Planck equation. 
# We start from the ESKC equation

# $$
# W(\mathbf{x},t|\mathbf{x}_0,t_0)  = \int_{-\infty}^{\infty} W(\mathbf{x},t|\mathbf{x}',t')W(\mathbf{x}',t'|\mathbf{x}_0,t_0)d\mathbf{x}'.
# $$

# Define $s=t'-t_0$, $\tau=t-t'$ and $t-t_0=s+\tau$. We have then

# $$
# W(\mathbf{x},s+\tau|\mathbf{x}_0)  = \int_{-\infty}^{\infty} W(\mathbf{x},\tau|\mathbf{x}')W(\mathbf{x}',s|\mathbf{x}_0)d\mathbf{x}'.
# $$

# Assume now that $\tau$ is very small so that we can make an expansion in terms of a small step $xi$, with $\mathbf{x}'=\mathbf{x}-\xi$, that is

# $$
# W(\mathbf{x},s|\mathbf{x}_0)+\frac{\partial W}{\partial s}\tau +O(\tau^2) = \int_{-\infty}^{\infty} W(\mathbf{x},\tau|\mathbf{x}-\xi)W(\mathbf{x}-\xi,s|\mathbf{x}_0)d\mathbf{x}'.
# $$

# We assume that $W(\mathbf{x},\tau|\mathbf{x}-\xi)$ takes non-negligible values only when $\xi$ is small. This is just another way of stating the Master equation!!
# 
# We say thus that $\mathbf{x}$ changes only by a small amount in the time interval $\tau$. 
# This means that we can make a Taylor expansion in terms of $\xi$, that is we
# expand

# $$
# W(\mathbf{x},\tau|\mathbf{x}-\xi)W(\mathbf{x}-\xi,s|\mathbf{x}_0) =
# \sum_{n=0}^{\infty}\frac{(-\xi)^n}{n!}\frac{\partial^n}{\partial x^n}\left[W(\mathbf{x}+\xi,\tau|\mathbf{x})W(\mathbf{x},s|\mathbf{x}_0)
# \right].
# $$

# We can then rewrite the ESKC equation as

# $$
# \frac{\partial W}{\partial s}\tau=-W(\mathbf{x},s|\mathbf{x}_0)+
# \sum_{n=0}^{\infty}\frac{(-\xi)^n}{n!}\frac{\partial^n}{\partial x^n}
# \left[W(\mathbf{x},s|\mathbf{x}_0)\int_{-\infty}^{\infty} \xi^nW(\mathbf{x}+\xi,\tau|\mathbf{x})d\xi\right].
# $$

# We have neglected higher powers of $\tau$ and have used that for $n=0$ 
# we get simply $W(\mathbf{x},s|\mathbf{x}_0)$ due to normalization.
# 
# We say thus that $\mathbf{x}$ changes only by a small amount in the time interval $\tau$. 
# This means that we can make a Taylor expansion in terms of $\xi$, that is we
# expand

# $$
# W(\mathbf{x},\tau|\mathbf{x}-\xi)W(\mathbf{x}-\xi,s|\mathbf{x}_0) =
# \sum_{n=0}^{\infty}\frac{(-\xi)^n}{n!}\frac{\partial^n}{\partial x^n}\left[W(\mathbf{x}+\xi,\tau|\mathbf{x})W(\mathbf{x},s|\mathbf{x}_0)
# \right].
# $$

# We can then rewrite the ESKC equation as

# $$
# \frac{\partial W(\mathbf{x},s|\mathbf{x}_0)}{\partial s}\tau=-W(\mathbf{x},s|\mathbf{x}_0)+
# \sum_{n=0}^{\infty}\frac{(-\xi)^n}{n!}\frac{\partial^n}{\partial x^n}
# \left[W(\mathbf{x},s|\mathbf{x}_0)\int_{-\infty}^{\infty} \xi^nW(\mathbf{x}+\xi,\tau|\mathbf{x})d\xi\right].
# $$

# We have neglected higher powers of $\tau$ and have used that for $n=0$ 
# we get simply $W(\mathbf{x},s|\mathbf{x}_0)$ due to normalization.
# 
# We simplify the above by introducing the moments

# $$
# M_n=\frac{1}{\tau}\int_{-\infty}^{\infty} \xi^nW(\mathbf{x}+\xi,\tau|\mathbf{x})d\xi=
# \frac{\langle [\Delta x(\tau)]^n\rangle}{\tau},
# $$

# resulting in

# $$
# \frac{\partial W(\mathbf{x},s|\mathbf{x}_0)}{\partial s}=
# \sum_{n=1}^{\infty}\frac{(-\xi)^n}{n!}\frac{\partial^n}{\partial x^n}
# \left[W(\mathbf{x},s|\mathbf{x}_0)M_n\right].
# $$

# When $\tau \rightarrow 0$ we assume that $\langle [\Delta x(\tau)]^n\rangle \rightarrow 0$ more rapidly than $\tau$ itself if $n > 2$. 
# When $\tau$ is much larger than the standard correlation time of 
# system then $M_n$ for $n > 2$ can normally be neglected.
# This means that fluctuations become negligible at large time scales.
# 
# If we neglect such terms we can rewrite the ESKC equation as

# $$
# \frac{\partial W(\mathbf{x},s|\mathbf{x}_0)}{\partial s}=
# -\frac{\partial M_1W(\mathbf{x},s|\mathbf{x}_0)}{\partial x}+
# \frac{1}{2}\frac{\partial^2 M_2W(\mathbf{x},s|\mathbf{x}_0)}{\partial x^2}.
# $$

# In a more compact form we have

# $$
# \frac{\partial W}{\partial s}=
# -\frac{\partial M_1W}{\partial x}+
# \frac{1}{2}\frac{\partial^2 M_2W}{\partial x^2},
# $$

# which is the Fokker-Planck equation!  It is trivial to replace 
# position with velocity (momentum).
# 
# Consider a particle  suspended in a liquid. On its path through the liquid it will continuously collide with the liquid molecules. Because on average the particle  will collide more often on the front side than on the back side, it will experience a systematic force proportional with its velocity, and directed opposite to its velocity. Besides this systematic force the particle  will experience a stochastic force  $\mathbf{F}(t)$. 
# The equations of motion are 
# * $\frac{d\mathbf{r}}{dt}=\mathbf{v}$ and 
# 
# * $\frac{d\mathbf{v}}{dt}=-\xi \mathbf{v}+\mathbf{F}$.
# 
# From hydrodynamics  we know that the friction constant  $\xi$ is given by

# $$
# \xi =6\pi \eta a/m
# $$

# where $\eta$ is the viscosity  of the solvent and a is the radius of the particle .
# 
# Solving the second equation in the previous slide we get

# $$
# \mathbf{v}(t)=\mathbf{v}_{0}e^{-\xi t}+\int_{0}^{t}d\tau e^{-\xi (t-\tau )}\mathbf{F }(\tau ).
# $$

# If we want to get some useful information out of this, we have to average over all possible realizations of 
# $\mathbf{F}(t)$, with the initial velocity as a condition. A useful quantity for example is

# $$
# \langle \mathbf{v}(t)\cdot \mathbf{v}(t)\rangle_{\mathbf{v}_{0}}=v_{0}^{-\xi 2t}
# +2\int_{0}^{t}d\tau e^{-\xi (2t-\tau)}\mathbf{v}_{0}\cdot \langle \mathbf{F}(\tau )\rangle_{\mathbf{v}_{0}}
# $$

# $$
# +\int_{0}^{t}d\tau ^{\prime }\int_{0}^{t}d\tau e^{-\xi (2t-\tau -\tau ^{\prime })}
# \langle \mathbf{F}(\tau )\cdot \mathbf{F}(\tau ^{\prime })\rangle_{ \mathbf{v}_{0}}.
# $$

# In order to continue we have to make some assumptions about the conditional averages of the stochastic forces. 
# In view of the chaotic character of the stochastic forces the following 
# assumptions seem to be appropriate

# $$
# \langle \mathbf{F}(t)\rangle=0,
# $$

# and

# $$
# \langle \mathbf{F}(t)\cdot \mathbf{F}(t^{\prime })\rangle_{\mathbf{v}_{0}}=  C_{\mathbf{v}_{0}}\delta (t-t^{\prime }).
# $$

# We omit the subscript $\mathbf{v}_{0}$, when the quantity of interest turns out to be independent of $\mathbf{v}_{0}$. Using the last three equations we get

# $$
# \langle \mathbf{v}(t)\cdot \mathbf{v}(t)\rangle_{\mathbf{v}_{0}}=v_{0}^{2}e^{-2\xi t}+\frac{C_{\mathbf{v}_{0}}}{2\xi }(1-e^{-2\xi t}).
# $$

# For large t this should be equal to 3kT/m, from which it follows that

# $$
# \langle \mathbf{F}(t)\cdot \mathbf{F}(t^{\prime })\rangle =6\frac{kT}{m}\xi \delta (t-t^{\prime }).
# $$

# This result is called the fluctuation-dissipation theorem .
# 
# Integrating

# $$
# \mathbf{v}(t)=\mathbf{v}_{0}e^{-\xi t}+\int_{0}^{t}d\tau e^{-\xi (t-\tau )}\mathbf{F }(\tau ),
# $$

# we get

# $$
# \mathbf{r}(t)=\mathbf{r}_{0}+\mathbf{v}_{0}\frac{1}{\xi }(1-e^{-\xi t})+
# \int_0^td\tau \int_0^{\tau}\tau ^{\prime } e^{-\xi (\tau -\tau ^{\prime })}\mathbf{F}(\tau ^{\prime }),
# $$

# from which we calculate the mean square displacement

# $$
# \langle ( \mathbf{r}(t)-\mathbf{r}_{0})^{2}\rangle _{\mathbf{v}_{0}}=\frac{v_0^2}{\xi}(1-e^{-\xi t})^{2}+\frac{3kT}{m\xi ^{2}}(2\xi t-3+4e^{-\xi t}-e^{-2\xi t}).
# $$

# For very large $t$ this becomes

# $$
# \langle (\mathbf{r}(t)-\mathbf{r}_{0})^{2}\rangle =\frac{6kT}{m\xi }t
# $$

# from which we get the Einstein relation

# $$
# D= \frac{kT}{m\xi }
# $$

# where we have used $\langle (\mathbf{r}(t)-\mathbf{r}_{0})^{2}\rangle =6Dt$.

# ## Code example for two electrons in a quantum dots

# In[5]:


# 2-electron VMC code for 2dim quantum dot with importance sampling
# Using gaussian rng for new positions and Metropolis- Hastings 
# No energy minimization
from math import exp, sqrt
from random import random, seed, normalvariate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from numba import jit,njit


#Read name of output file from command line
if len(sys.argv) == 2:
    outfilename = sys.argv[1]
else:
    print('\nError: Name of output file must be given as command line argument.\n')
outfile = open(outfilename,'w')

# Trial wave function for the 2-electron quantum dot in two dims
def WaveFunction(r,alpha,beta):
    r1 = r[0,0]**2 + r[0,1]**2
    r2 = r[1,0]**2 + r[1,1]**2
    r12 = sqrt((r[0,0]-r[1,0])**2 + (r[0,1]-r[1,1])**2)
    deno = r12/(1+beta*r12)
    return exp(-0.5*alpha*(r1+r2)+deno)

# Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
def LocalEnergy(r,alpha,beta):
    
    r1 = (r[0,0]**2 + r[0,1]**2)
    r2 = (r[1,0]**2 + r[1,1]**2)
    r12 = sqrt((r[0,0]-r[1,0])**2 + (r[0,1]-r[1,1])**2)
    deno = 1.0/(1+beta*r12)
    deno2 = deno*deno
    return 0.5*(1-alpha*alpha)*(r1 + r2) +2.0*alpha + 1.0/r12+deno2*(alpha*r12-deno2+2*beta*deno-1.0/r12)

# Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
def QuantumForce(r,alpha,beta):

    qforce = np.zeros((NumberParticles,Dimension), np.double)
    r12 = sqrt((r[0,0]-r[1,0])**2 + (r[0,1]-r[1,1])**2)
    deno = 1.0/(1+beta*r12)
    qforce[0,:] = -2*r[0,:]*alpha*(r[0,:]-r[1,:])*deno*deno/r12
    qforce[1,:] = -2*r[1,:]*alpha*(r[1,:]-r[0,:])*deno*deno/r12
    return qforce
    
# The Monte Carlo sampling with the Metropolis algo
# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit()
def MonteCarloSampling():

    NumberMCcycles= 100000
    # Parameters in the Fokker-Planck simulation of the quantum force
    D = 0.5
    TimeStep = 0.05
    # positions
    PositionOld = np.zeros((NumberParticles,Dimension), np.double)
    PositionNew = np.zeros((NumberParticles,Dimension), np.double)
    # Quantum force
    QuantumForceOld = np.zeros((NumberParticles,Dimension), np.double)
    QuantumForceNew = np.zeros((NumberParticles,Dimension), np.double)

    # seed for rng generator 
    seed()
    # start variational parameter  loops, two parameters here
    alpha = 0.9
    for ia in range(MaxVariations):
        alpha += .025
        AlphaValues[ia] = alpha
        beta = 0.2 
        for jb in range(MaxVariations):
            beta += .01
            BetaValues[jb] = beta
            energy = energy2 = 0.0
            DeltaE = 0.0
            #Initial position
            for i in range(NumberParticles):
                for j in range(Dimension):
                    PositionOld[i,j] = normalvariate(0.0,1.0)*sqrt(TimeStep)
            wfold = WaveFunction(PositionOld,alpha,beta)
            QuantumForceOld = QuantumForce(PositionOld,alpha, beta)

            #Loop over MC MCcycles
            for MCcycle in range(NumberMCcycles):
                #Trial position moving one particle at the time
                for i in range(NumberParticles):
                    for j in range(Dimension):
                        PositionNew[i,j] = PositionOld[i,j]+normalvariate(0.0,1.0)*sqrt(TimeStep)+                                           QuantumForceOld[i,j]*TimeStep*D
                    wfnew = WaveFunction(PositionNew,alpha,beta)
                    QuantumForceNew = QuantumForce(PositionNew,alpha, beta)
                    GreensFunction = 0.0
                    for j in range(Dimension):
                        GreensFunction += 0.5*(QuantumForceOld[i,j]+QuantumForceNew[i,j])*	                              (D*TimeStep*0.5*(QuantumForceOld[i,j]-QuantumForceNew[i,j])-                                      PositionNew[i,j]+PositionOld[i,j])
      
                    GreensFunction = exp(GreensFunction)
                    ProbabilityRatio = GreensFunction*wfnew**2/wfold**2
                    #Metropolis-Hastings test to see whether we accept the move
                    if random() <= ProbabilityRatio:
                       for j in range(Dimension):
                           PositionOld[i,j] = PositionNew[i,j]
                           QuantumForceOld[i,j] = QuantumForceNew[i,j]
                       wfold = wfnew
                DeltaE = LocalEnergy(PositionOld,alpha,beta)
                energy += DeltaE
                energy2 += DeltaE**2
            # We calculate mean, variance and error (no blocking applied)
            energy /= NumberMCcycles
            energy2 /= NumberMCcycles
            variance = energy2 - energy**2
            error = sqrt(variance/NumberMCcycles)
            Energies[ia,jb] = energy    
            outfile.write('%f %f %f %f %f\n' %(alpha,beta,energy,variance,error))
    return Energies, AlphaValues, BetaValues


#Here starts the main program with variable declarations
NumberParticles = 2
Dimension = 2
MaxVariations = 10
Energies = np.zeros((MaxVariations,MaxVariations))
AlphaValues = np.zeros(MaxVariations)
BetaValues = np.zeros(MaxVariations)
(Energies, AlphaValues, BetaValues) = MonteCarloSampling()
outfile.close()
# Prepare for plots
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
X, Y = np.meshgrid(AlphaValues, BetaValues)
surf = ax.plot_surface(X, Y, Energies,cmap=cm.coolwarm,linewidth=0, antialiased=False)
# Customize the z axis.
zmin = np.matrix(Energies).min()
zmax = np.matrix(Energies).max()
ax.set_zlim(zmin, zmax)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$')
ax.set_zlabel(r'$\langle E \rangle$')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# ### Bringing the gradient optmization
# 
# The simple one-particle case in a harmonic oscillator trap

# In[6]:


# Gradient descent stepping with analytical derivative
import numpy as np
from scipy.optimize import minimize
def DerivativeE(x):
    return x-1.0/(4*x*x*x);

def Energy(x):
   return x*x*0.5+1.0/(8*x*x);
x0 = 1.0
eta = 0.1
Niterations = 100

for iter in range(Niterations):
    gradients = DerivativeE(x0)
    x0 -= eta*gradients

print(x0)


# In[7]:


# 2-electron VMC code for 2dim quantum dot with importance sampling
# Using gaussian rng for new positions and Metropolis- Hastings 
from math import exp, sqrt
from random import random, seed, normalvariate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from numba import jit


# Trial wave function for the 2-electron quantum dot in two dims
def WaveFunction(r,alpha):
    r1 = r[0,0]**2 + r[0,1]**2
    r2 = r[1,0]**2 + r[1,1]**2
    return exp(-0.5*alpha*(r1+r2))

# Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
def LocalEnergy(r,alpha):
    
    r1 = (r[0,0]**2 + r[0,1]**2)
    r2 = (r[1,0]**2 + r[1,1]**2)
    return 0.5*(1-alpha*alpha)*(r1 + r2) +2.0*alpha

# Derivate of wave function ansatz as function of variational parameters
def DerivativeWFansatz(r,alpha):
    
    r1 = (r[0,0]**2 + r[0,1]**2)
    r2 = (r[1,0]**2 + r[1,1]**2)
    WfDer = -(r1+r2)
    return  WfDer

# Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
def QuantumForce(r,alpha):

    qforce = np.zeros((NumberParticles,Dimension), np.double)
    qforce[0,:] = -2*r[0,:]*alpha
    qforce[1,:] = -2*r[1,:]*alpha
    return qforce
    
# Computing the derivative of the energy and the energy 
# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@jit
def EnergyMinimization(alpha):

    NumberMCcycles= 1000
    # Parameters in the Fokker-Planck simulation of the quantum force
    D = 0.5
    TimeStep = 0.05
    # positions
    PositionOld = np.zeros((NumberParticles,Dimension), np.double)
    PositionNew = np.zeros((NumberParticles,Dimension), np.double)
    # Quantum force
    QuantumForceOld = np.zeros((NumberParticles,Dimension), np.double)
    QuantumForceNew = np.zeros((NumberParticles,Dimension), np.double)

    # seed for rng generator 
    seed()
    energy = 0.0
    DeltaE = 0.0
    EnergyDer = 0.0
    DeltaPsi = 0.0
    DerivativePsiE = 0.0
    #Initial position
    for i in range(NumberParticles):
        for j in range(Dimension):
            PositionOld[i,j] = normalvariate(0.0,1.0)*sqrt(TimeStep)
    wfold = WaveFunction(PositionOld,alpha)
    QuantumForceOld = QuantumForce(PositionOld,alpha)

    #Loop over MC MCcycles
    for MCcycle in range(NumberMCcycles):
        #Trial position moving one particle at the time
        for i in range(NumberParticles):
            for j in range(Dimension):
                PositionNew[i,j] = PositionOld[i,j]+normalvariate(0.0,1.0)*sqrt(TimeStep)+                                       QuantumForceOld[i,j]*TimeStep*D
            wfnew = WaveFunction(PositionNew,alpha)
            QuantumForceNew = QuantumForce(PositionNew,alpha)
            GreensFunction = 0.0
            for j in range(Dimension):
                GreensFunction += 0.5*(QuantumForceOld[i,j]+QuantumForceNew[i,j])*	                              (D*TimeStep*0.5*(QuantumForceOld[i,j]-QuantumForceNew[i,j])-                                      PositionNew[i,j]+PositionOld[i,j])
      
            GreensFunction = exp(GreensFunction)
            ProbabilityRatio = GreensFunction*wfnew**2/wfold**2
            #Metropolis-Hastings test to see whether we accept the move
            if random() <= ProbabilityRatio:
                for j in range(Dimension):
                    PositionOld[i,j] = PositionNew[i,j]
                    QuantumForceOld[i,j] = QuantumForceNew[i,j]
                wfold = wfnew
        DeltaE = LocalEnergy(PositionOld,alpha)
        DeltaPsi = DerivativeWFansatz(PositionOld,alpha)
        energy += DeltaE
        DerivativePsiE += DeltaPsi*DeltaE
            
    # We calculate mean, variance and error (no blocking applied)
    energy /= NumberMCcycles
    DerivativePsiE /= NumberMCcycles
    DeltaPsi /= NumberMCcycles
    EnergyDer  = 2*(DerivativePsiE-DeltaPsi*energy)
    return energy, EnergyDer


#Here starts the main program with variable declarations
NumberParticles = 2
Dimension = 2
# guess for variational parameters
x0 = 1.5
# Set up iteration using stochastic gradient method
Energy =0 ; EnergyDer = 0
Energy, EnergyDer = EnergyMinimization(x0)
print(Energy, EnergyDer)

eta = 0.01
Niterations = 100

for iter in range(Niterations):
    gradients = EnergyDer
    x0 -= eta*gradients
    Energy, EnergyDer = EnergyMinimization(x0)

print(x0)


# ## VMC for fermions: Efficient calculation of Slater determinants
# The potentially most time-consuming part is the
# evaluation of the gradient and the Laplacian of an $N$-particle  Slater
# determinant. 
# 
# We have to differentiate the determinant with respect to
# all spatial coordinates of all particles. A brute force
# differentiation would involve $N\cdot d$ evaluations of the entire
# determinant which would even worsen the already undesirable time
# scaling, making it $Nd\cdot O(N^3)\sim O(d\cdot N^4)$.
# 
# This poses serious hindrances to the overall efficiency of our code.
# 
# The efficiency can be improved however if we move only one electron at the time.
# The Slater determinant matrix $\hat{D}$ is defined by the matrix elements

# $$
# d_{ij}=\phi_j(x_i)
# $$

# where $\phi_j(\mathbf{r}_i)$ is a single particle  wave function.
# The columns correspond to the position of a given particle 
# while the rows stand for the various quantum numbers.
# 
# What we need to realize is that when differentiating a Slater
# determinant with respect to some given coordinate, only one row of the
# corresponding Slater matrix is changed. 
# 
# Therefore, by recalculating
# the whole determinant we risk producing redundant information. The
# solution turns out to be an algorithm that requires to keep track of
# the *inverse* of the Slater matrix.
# 
# Let the current position in phase space be represented by the $(N\cdot d)$-element 
# vector $\mathbf{r}^{\mathrm{old}}$ and the new suggested
# position by the vector $\mathbf{r}^{\mathrm{new}}$.
# 
# The inverse of $\hat{D}$ can be expressed in terms of its
# cofactors $C_{ij}$ and its determinant (this our notation for a determinant) $\vert\hat{D}\vert$:

# <!-- Equation labels as ordinary links -->
# <div id="eq:inverse_cofactor"></div>
# 
# $$
# \begin{equation}
# d_{ij}^{-1} = \frac{C_{ji}}{\vert\hat{D}\vert}
# \label{eq:inverse_cofactor} \tag{12}
# \end{equation}
# $$

# Notice that the interchanged indices indicate that the matrix of cofactors is to be transposed.
# 
# If $\hat{D}$ is invertible, then we must obviously have $\hat{D}^{-1}\hat{D}= \mathbf{1}$, or explicitly in terms of the individual
# elements of $\hat{D}$ and $\hat{D}^{-1}$:

# <!-- Equation labels as ordinary links -->
# <div id="eq:unity_explicitely"></div>
# 
# $$
# \begin{equation}
# \sum_{k=1}^N d_{ik}^{\phantom X}d_{kj}^{-1} = \delta_{ij}^{\phantom X}
# \label{eq:unity_explicitely} \tag{13}
# \end{equation}
# $$

# Consider the ratio, which we shall call $R$, between $\vert\hat{D}(\mathbf{r}^{\mathrm{new}})\vert$ and $\vert\hat{D}(\mathbf{r}^{\mathrm{old}})\vert$. 
# By definition, each of these determinants can
# individually be expressed in terms of the *i*-th row of its cofactor
# matrix

# <!-- Equation labels as ordinary links -->
# <div id="eq:detratio_cofactors"></div>
# 
# $$
# \begin{equation}
# R\equiv\frac{\vert\hat{D}(\mathbf{r}^{\mathrm{new}})\vert}
# {\vert\hat{D}(\mathbf{r}^{\mathrm{old}})\vert} =
# \frac{\sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{new}})\,
# C_{ij}(\mathbf{r}^{\mathrm{new}})}
# {\sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{old}})\,
# C_{ij}(\mathbf{r}^{\mathrm{old}})}
# \label{eq:detratio_cofactors} \tag{14}
# \end{equation}
# $$

# Suppose now that we move only one particle  at a time, meaning that
# $\mathbf{r}^{\mathrm{new}}$ differs from $\mathbf{r}^{\mathrm{old}}$ by the
# position of only one, say the *i*-th, particle . This means that $\hat{D}(\mathbf{r}^{\mathrm{new}})$ and $\hat{D}(\mathbf{r}^{\mathrm{old}})$ differ
# only by the entries of the *i*-th row.  Recall also that the *i*-th row
# of a cofactor matrix $\hat{C}$ is independent of the entries of the
# *i*-th row of its corresponding matrix $\hat{D}$. In this particular
# case we therefore get that the *i*-th row of $\hat{C}(\mathbf{r}^{\mathrm{new}})$ 
# and $\hat{C}(\mathbf{r}^{\mathrm{old}})$ must be
# equal. Explicitly, we have:

# <!-- Equation labels as ordinary links -->
# <div id="_auto5"></div>
# 
# $$
# \begin{equation}
# C_{ij}(\mathbf{r}^{\mathrm{new}}) = C_{ij}(\mathbf{r}^{\mathrm{old}})\quad
# \forall\ j\in\{1,\dots,N\}
# \label{_auto5} \tag{15}
# \end{equation}
# $$

# Inserting this into the numerator of eq. ([14](#eq:detratio_cofactors))
# and using eq. ([12](#eq:inverse_cofactor)) to substitute the cofactors
# with the elements of the inverse matrix, we get:

# <!-- Equation labels as ordinary links -->
# <div id="_auto6"></div>
# 
# $$
# \begin{equation}
# R =\frac{\sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{new}})\,
# C_{ij}(\mathbf{r}^{\mathrm{old}})}
# {\sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{old}})\,
# C_{ij}(\mathbf{r}^{\mathrm{old}})} =
# \frac{\sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{new}})\,
# d_{ji}^{-1}(\mathbf{r}^{\mathrm{old}})}
# {\sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{old}})\,
# d_{ji}^{-1}(\mathbf{r}^{\mathrm{old}})}
# \label{_auto6} \tag{16}
# \end{equation}
# $$

# Now by eq. ([13](#eq:unity_explicitely)) the denominator of the rightmost
# expression must be unity, so that we finally arrive at:

# <!-- Equation labels as ordinary links -->
# <div id="eq:detratio_inverse"></div>
# 
# $$
# \begin{equation}
# R =
# \sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{new}})\,
# d_{ji}^{-1}(\mathbf{r}^{\mathrm{old}}) = 
# \sum_{j=1}^N \phi_j(\mathbf{r}_i^{\mathrm{new}})\,
# d_{ji}^{-1}(\mathbf{r}^{\mathrm{old}})
# \label{eq:detratio_inverse} \tag{17}
# \end{equation}
# $$

# What this means is that in order to get the ratio when only the *i*-th
# particle  has been moved, we only need to calculate the dot
# product of the vector $\left(\phi_1(\mathbf{r}_i^\mathrm{new}),\,\dots,\,\phi_N(\mathbf{r}_i^\mathrm{new})\right)$ of single particle  wave functions
# evaluated at this new position with the *i*-th column of the inverse
# matrix $\hat{D}^{-1}$ evaluated at the original position. Such
# an operation has a time scaling of $O(N)$. The only extra thing we
# need to do is to maintain the inverse matrix $\hat{D}^{-1}(\mathbf{x}^{\mathrm{old}})$.
# 
# If the new position $\mathbf{r}^{\mathrm{new}}$ is accepted, then the
# inverse matrix can by suitably updated by an algorithm having a time
# scaling of $O(N^2)$.  This algorithm goes as
# follows. First we update all but the *i*-th column of $\hat{D}^{-1}$. For each column $j\neq i$, we first calculate the quantity:

# <!-- Equation labels as ordinary links -->
# <div id="eq:inverse_update_1"></div>
# 
# $$
# \begin{equation}
# S_j =
# (\hat{D}(\mathbf{r}^{\mathrm{new}})\times
# \hat{D}^{-1}(\mathbf{r}^{\mathrm{old}}))_{ij} =
# \sum_{l=1}^N d_{il}(\mathbf{r}^{\mathrm{new}})\,
# d^{-1}_{lj}(\mathbf{r}^{\mathrm{old}})
# \label{eq:inverse_update_1} \tag{18}
# \end{equation}
# $$

# The new elements of the *j*-th column of $\hat{D}^{-1}$ are then given
# by:

# <!-- Equation labels as ordinary links -->
# <div id="eq:inverse_update_2"></div>
# 
# $$
# \begin{equation}
# d_{kj}^{-1}(\mathbf{r}^{\mathrm{new}}) =
# d_{kj}^{-1}(\mathbf{r}^{\mathrm{old}}) -
# \frac{S_j}{R}\,d_{ki}^{-1}(\mathbf{r}^{\mathrm{old}})\quad
# \begin{array}{ll}
# \forall\ \ k\in\{1,\dots,N\}\\j\neq i
# \end{array}
# \label{eq:inverse_update_2} \tag{19}
# \end{equation}
# $$

# Finally the elements of the *i*-th column of $\hat{D}^{-1}$ are updated
# simply as follows:

# <!-- Equation labels as ordinary links -->
# <div id="eq:inverse_update_3"></div>
# 
# $$
# \begin{equation}
# d_{ki}^{-1}(\mathbf{r}^{\mathrm{new}}) =
# \frac{1}{R}\,d_{ki}^{-1}(\mathbf{r}^{\mathrm{old}})\quad
# \forall\ \ k\in\{1,\dots,N\}
# \label{eq:inverse_update_3} \tag{20}
# \end{equation}
# $$

# We see from these formulas that the time scaling of an update of
# $\hat{D}^{-1}$ after changing one row of $\hat{D}$ is $O(N^2)$.
# 
# The scheme is also applicable for the calculation of the ratios
# involving derivatives. It turns
# out that differentiating the Slater determinant with respect
# to the coordinates of a single particle  $\mathbf{r}_i$ changes only the
# *i*-th row of the corresponding Slater matrix.

# ### The gradient and the Laplacian
# 
# The gradient and the Laplacian can therefore be calculated as follows:

# $$
# \frac{\vec\nabla_i\vert\hat{D}(\mathbf{r})\vert}{\vert\hat{D}(\mathbf{r})\vert} =
# \sum_{j=1}^N \vec\nabla_i d_{ij}(\mathbf{r})d_{ji}^{-1}(\mathbf{r}) =
# \sum_{j=1}^N \vec\nabla_i \phi_j(\mathbf{r}_i)d_{ji}^{-1}(\mathbf{r})
# $$

# and

# $$
# \frac{\nabla^2_i\vert\hat{D}(\mathbf{r})\vert}{\vert\hat{D}(\mathbf{r})\vert} =
# \sum_{j=1}^N \nabla^2_i d_{ij}(\mathbf{r})d_{ji}^{-1}(\mathbf{r}) =
# \sum_{j=1}^N \nabla^2_i \phi_j(\mathbf{r}_i)\,d_{ji}^{-1}(\mathbf{r})
# $$

# Thus, to calculate all the derivatives of the Slater determinant, we
# only need the derivatives of the single particle  wave functions
# ($\vec\nabla_i \phi_j(\mathbf{r}_i)$ and $\nabla^2_i \phi_j(\mathbf{r}_i)$)
# and the elements of the corresponding inverse Slater matrix ($\hat{D}^{-1}(\mathbf{r}_i)$). A calculation of a single derivative is by the
# above result an $O(N)$ operation. Since there are $d\cdot N$
# derivatives, the time scaling of the total evaluation becomes
# $O(d\cdot N^2)$. With an $O(N^2)$ updating algorithm for the
# inverse matrix, the total scaling is no worse, which is far better
# than the brute force approach yielding $O(d\cdot N^4)$.
# 
# **Important note**: In most cases you end with closed form expressions for the single-particle  wave functions. It is then useful to calculate the various derivatives and make separate functions
# for them.
# 
# The Slater determinant takes the form

# $$
# \Phi(\mathbf{r}_1,\mathbf{r}_2,,\mathbf{r}_3,\mathbf{r}_4, \alpha,\beta,\gamma,\delta)=\frac{1}{\sqrt{4!}}
# \left| \begin{array}{cccc} \psi_{100\uparrow}(\mathbf{r}_1)& \psi_{100\uparrow}(\mathbf{r}_2)& \psi_{100\uparrow}(\mathbf{r}_3)&\psi_{100\uparrow}(\mathbf{r}_4) \\
# \psi_{100\downarrow}(\mathbf{r}_1)& \psi_{100\downarrow}(\mathbf{r}_2)& \psi_{100\downarrow}(\mathbf{r}_3)&\psi_{100\downarrow}(\mathbf{r}_4) \\
# \psi_{200\uparrow}(\mathbf{r}_1)& \psi_{200\uparrow}(\mathbf{r}_2)& \psi_{200\uparrow}(\mathbf{r}_3)&\psi_{200\uparrow}(\mathbf{r}_4) \\
# \psi_{200\downarrow}(\mathbf{r}_1)& \psi_{200\downarrow}(\mathbf{r}_2)& \psi_{200\downarrow}(\mathbf{r}_3)&\psi_{200\downarrow}(\mathbf{r}_4) \end{array} \right|.
# $$

# The Slater determinant as written is zero since the spatial wave functions for the spin up and spin down 
# states are equal.  
# But we can rewrite it as the product of two Slater determinants, one for spin up and one for spin down.
# 
# We can rewrite it as

# $$
# \Phi(\mathbf{r}_1,\mathbf{r}_2,,\mathbf{r}_3,\mathbf{r}_4, \alpha,\beta,\gamma,\delta)=\det\uparrow(1,2)\det\downarrow(3,4)-\det\uparrow(1,3)\det\downarrow(2,4)
# $$

# $$
# -\det\uparrow(1,4)\det\downarrow(3,2)+\det\uparrow(2,3)\det\downarrow(1,4)-\det\uparrow(2,4)\det\downarrow(1,3)
# $$

# $$
# +\det\uparrow(3,4)\det\downarrow(1,2),
# $$

# where we have defined

# $$
# \det\uparrow(1,2)=\frac{1}{\sqrt{2}}\left| \begin{array}{cc} \psi_{100\uparrow}(\mathbf{r}_1)& \psi_{100\uparrow}(\mathbf{r}_2)\\
# \psi_{200\uparrow}(\mathbf{r}_1)& \psi_{200\uparrow}(\mathbf{r}_2) \end{array} \right|,
# $$

# and

# $$
# \det\downarrow(3,4)=\frac{1}{\sqrt{2}}\left| \begin{array}{cc} \psi_{100\downarrow}(\mathbf{r}_3)& \psi_{100\downarrow}(\mathbf{r}_4)\\
# \psi_{200\downarrow}(\mathbf{r}_3)& \psi_{200\downarrow}(\mathbf{r}_4) \end{array} \right|.
# $$

# We want to avoid to sum over spin variables, in particular when the interaction does not depend on spin.
# 
# It can be shown, see for example Moskowitz and Kalos, [Int. J. Quantum Chem. **20** 1107 (1981)](http://onlinelibrary.wiley.com/doi/10.1002/qua.560200508/abstract), that for the variational energy
# we can approximate the Slater determinant as

# $$
# \Phi(\mathbf{r}_1,\mathbf{r}_2,,\mathbf{r}_3,\mathbf{r}_4, \alpha,\beta,\gamma,\delta) \propto \det\uparrow(1,2)\det\downarrow(3,4),
# $$

# or more generally as

# $$
# \Phi(\mathbf{r}_1,\mathbf{r}_2,\dots \mathbf{r}_N) \propto \det\uparrow \det\downarrow,
# $$

# where we have the Slater determinant as the product of a spin up part involving the number of electrons with spin up only (2 for beryllium and 5 for neon) and a spin down part involving the electrons with spin down.
# 
# This ansatz is not antisymmetric under the exchange of electrons with  opposite spins but it can be shown (show this) that it gives the same
# expectation value for the energy as the full Slater determinant.
# 
# As long as the Hamiltonian is spin independent, the above is correct. It is rather straightforward to see this if you go back to the equations for the energy discussed earlier  this semester.
# 
# We will thus
# factorize the full determinant $\vert\hat{D}\vert$ into two smaller ones, where 
# each can be identified with $\uparrow$ and $\downarrow$
# respectively:

# $$
# \vert\hat{D}\vert = \vert\hat{D}\vert_\uparrow\cdot \vert\hat{D}\vert_\downarrow
# $$

# The combined dimensionality of the two smaller determinants equals the
# dimensionality of the full determinant. Such a factorization is
# advantageous in that it makes it possible to perform the calculation
# of the ratio $R$ and the updating of the inverse matrix separately for
# $\vert\hat{D}\vert_\uparrow$ and $\vert\hat{D}\vert_\downarrow$:

# $$
# \frac{\vert\hat{D}\vert^\mathrm{new}}{\vert\hat{D}\vert^\mathrm{old}} =
# \frac{\vert\hat{D}\vert^\mathrm{new}_\uparrow}
# {\vert\hat{D}\vert^\mathrm{old}_\uparrow}\cdot
# \frac{\vert\hat{D}\vert^\mathrm{new}_\downarrow
# }{\vert\hat{D}\vert^\mathrm{old}_\downarrow}
# $$

# This reduces the calculation time by a constant factor. The maximal
# time reduction happens in a system of equal numbers of $\uparrow$ and
# $\downarrow$ particles, so that the two factorized determinants are
# half the size of the original one.
# 
# Consider the case of moving only one particle  at a time which
# originally had the following time scaling for one transition:

# $$
# O_R(N)+O_\mathrm{inverse}(N^2)
# $$

# For the factorized determinants one of the two determinants is
# obviously unaffected by the change so that it cancels from the ratio
# $R$. 
# 
# Therefore, only one determinant of size $N/2$ is involved in each
# calculation of $R$ and update of the inverse matrix. The scaling of
# each transition then becomes:

# $$
# O_R(N/2)+O_\mathrm{inverse}(N^2/4)
# $$

# and the time scaling when the transitions for all $N$ particles are
# put together:

# $$
# O_R(N^2/2)+O_\mathrm{inverse}(N^3/4)
# $$

# which gives the same reduction as in the case of moving all particles
# at once.
# 
# Computing the ratios discussed above requires that we maintain 
# the inverse of the Slater matrix evaluated at the current position. 
# Each time a trial position is accepted, the row number $i$ of the Slater 
# matrix changes and updating its inverse has to be carried out. 
# Getting the inverse of an $N \times N$ matrix by Gaussian elimination has a 
# complexity of order of $\mathcal{O}(N^3)$ operations, a luxury that we 
# cannot afford for each time a particle  move is accepted.
# We will use the expression

# <!-- Equation labels as ordinary links -->
# <div id="updatingInverse"></div>
# 
# $$
# \label{updatingInverse} \tag{21}
# d^{-1}_{kj}(\mathbf{x^{new}}) = \left\{\begin{array}{l l}
#   d^{-1}_{kj}(\mathbf{x^{old}}) - \frac{d^{-1}_{ki}(\mathbf{x^{old}})}{R} \sum_{l=1}^{N} d_{il}(\mathbf{x^{new}})  d^{-1}_{lj}(\mathbf{x^{old}}) & \mbox{if $j \neq i$}\nonumber \\ \\
#  \frac{d^{-1}_{ki}(\mathbf{x^{old}})}{R} \sum_{l=1}^{N} d_{il}(\mathbf{x^{old}}) d^{-1}_{lj}(\mathbf{x^{old}}) & \mbox{if $j=i$}
# \end{array} \right.
# $$

# This equation scales as $O(N^2)$.
# The evaluation of the determinant of an $N \times N$ matrix by standard Gaussian elimination 
# requires $\mathbf{O}(N^3)$
# calculations. 
# As there are $Nd$ independent coordinates we need to evaluate $Nd$ Slater determinants 
# for the gradient (quantum force) and $Nd$ for the Laplacian (kinetic energy). 
# With the updating algorithm we need only to invert the Slater 
# determinant matrix once. This can be done by standard LU decomposition methods.

# ### Expectation value of the kinetic energy
# 
# The expectation value of the kinetic energy expressed in atomic units for electron $i$ is

# $$
# \langle \hat{K}_i \rangle = -\frac{1}{2}\frac{\langle\Psi|\nabla_{i}^2|\Psi \rangle}{\langle\Psi|\Psi \rangle},
# $$

# <!-- Equation labels as ordinary links -->
# <div id="kineticE"></div>
# 
# $$
# \begin{equation}
# \label{kineticE} \tag{22}
# K_i = -\frac{1}{2}\frac{\nabla_{i}^{2} \Psi}{\Psi}.
# \end{equation}
# $$

# $$
# \frac{\nabla^2 \Psi}{\Psi}  =  \frac{\nabla^2 ({\Psi_{D} \,  \Psi_C})}{\Psi_{D} \,  \Psi_C} = \frac{\nabla  \cdot [\nabla  {(\Psi_{D} \,  \Psi_C)}]}{\Psi_{D} \,  \Psi_C} = \frac{\nabla  \cdot [ \Psi_C \nabla  \Psi_{D} + \Psi_{D} \nabla   \Psi_C]}{\Psi_{D} \,  \Psi_C}\nonumber
# $$

# $$
# =  \frac{\nabla   \Psi_C \cdot \nabla  \Psi_{D} +  \Psi_C \nabla^2 \Psi_{D} + \nabla  \Psi_{D} \cdot \nabla   \Psi_C + \Psi_{D} \nabla^2  \Psi_C}{\Psi_{D} \,  \Psi_C}\nonumber
# $$

# <!-- Equation labels as ordinary links -->
# <div id="_auto7"></div>
# 
# $$
# \begin{equation} 
# \label{_auto7} \tag{23}
# \end{equation}
# $$

# <!-- Equation labels as ordinary links -->
# <div id="_auto8"></div>
# 
# $$
# \begin{equation}
# \frac{\nabla^2 \Psi}{\Psi}
#  =  \frac{\nabla^2 \Psi_{D}}{\Psi_{D}} + \frac{\nabla^2  \Psi_C}{ \Psi_C} + 2 \frac{\nabla  \Psi_{D}}{\Psi_{D}}\cdot\frac{\nabla   \Psi_C}{ \Psi_C}
# \label{_auto8} \tag{24}
# \end{equation}
# $$

# The second derivative of the Jastrow factor divided by the Jastrow factor (the way it enters the kinetic energy) is

# $$
# \left[\frac{\nabla^2 \Psi_C}{\Psi_C}\right]_x =\  
# 2\sum_{k=1}^{N}
# \sum_{i=1}^{k-1}\frac{\partial^2 g_{ik}}{\partial x_k^2}\ +\ 
# \sum_{k=1}^N
# \left(
# \sum_{i=1}^{k-1}\frac{\partial g_{ik}}{\partial x_k} -
# \sum_{i=k+1}^{N}\frac{\partial g_{ki}}{\partial x_i}
# \right)^2
# $$

# But we have a simple form for the function, namely

# $$
# \Psi_{C}=\prod_{i< j}\exp{f(r_{ij})}= \exp{\left\{\sum_{i<j}\frac{ar_{ij}}{1+\beta r_{ij}}\right\}},
# $$

# and it is easy to see that for particle  $k$
# we have

# $$
# \frac{\nabla^2_k \Psi_C}{\Psi_C }=
# \sum_{ij\ne k}\frac{(\mathbf{r}_k-\mathbf{r}_i)(\mathbf{r}_k-\mathbf{r}_j)}{r_{ki}r_{kj}}f'(r_{ki})f'(r_{kj})+
# \sum_{j\ne k}\left( f''(r_{kj})+\frac{2}{r_{kj}}f'(r_{kj})\right)
# $$

# Using

# $$
# f(r_{ij})= \frac{ar_{ij}}{1+\beta r_{ij}},
# $$

# and $g'(r_{kj})=dg(r_{kj})/dr_{kj}$ and 
# $g''(r_{kj})=d^2g(r_{kj})/dr_{kj}^2$  we find that for particle  $k$
# we have

# $$
# \frac{\nabla^2_k \Psi_C}{\Psi_C }=
# \sum_{ij\ne k}\frac{(\mathbf{r}_k-\mathbf{r}_i)(\mathbf{r}_k-\mathbf{r}_j)}{r_{ki}r_{kj}}\frac{a}{(1+\beta r_{ki})^2}
# \frac{a}{(1+\beta r_{kj})^2}+
# \sum_{j\ne k}\left(\frac{2a}{r_{kj}(1+\beta r_{kj})^2}-\frac{2a\beta}{(1+\beta r_{kj})^3}\right)
# $$

# The gradient and
# Laplacian can be calculated as follows:

# $$
# \frac{\mathbf{\nabla}_i\vert\hat{D}(\mathbf{r})\vert}
# {\vert\hat{D}(\mathbf{r})\vert} =
# \sum_{j=1}^N \vec\nabla_i d_{ij}(\mathbf{r})\,
# d_{ji}^{-1}(\mathbf{r}) =
# \sum_{j=1}^N \vec\nabla_i \phi_j(\mathbf{r}_i)\,
# d_{ji}^{-1}(\mathbf{r})
# $$

# and

# $$
# \frac{\nabla^2_i\vert\hat{D}(\mathbf{r})\vert}
# {\vert\hat{D}(\mathbf{r})\vert} =
# \sum_{j=1}^N \nabla^2_i d_{ij}(\mathbf{r})\,
# d_{ji}^{-1}(\mathbf{r}) =
# \sum_{j=1}^N \nabla^2_i \phi_j(\mathbf{r}_i)\,
# d_{ji}^{-1}(\mathbf{r})
# $$

# The gradient for the determinant is

# $$
# \frac{\mathbf{\nabla}_i\vert\hat{D}(\mathbf{r})\vert}
# {\vert\hat{D}(\mathbf{r})\vert} =
# \sum_{j=1}^N \mathbf{\nabla}_i d_{ij}(\mathbf{r})\,
# d_{ji}^{-1}(\mathbf{r}) =
# \sum_{j=1}^N \mathbf{\nabla}_i \phi_j(\mathbf{r}_i)\,
# d_{ji}^{-1}(\mathbf{r}).
# $$

# We have

# $$
# \Psi_C=\prod_{i< j}g(r_{ij})= \exp{\left\{\sum_{i<j}\frac{ar_{ij}}{1+\beta r_{ij}}\right\}},
# $$

# the gradient needed for the quantum force and local energy is easy to compute.  
# We get for particle  $k$

# $$
# \frac{ \nabla_k \Psi_C}{ \Psi_C }= \sum_{j\ne k}\frac{\mathbf{r}_{kj}}{r_{kj}}\frac{a}{(1+\beta r_{kj})^2},
# $$

# which is rather easy to code.  Remember to sum over all particles  when you compute the local energy.
# 
# We need to compute the ratio between wave functions, in particular  for the Slater determinants.

# $$
# R =\sum_{j=1}^N d_{ij}(\mathbf{r}^{\mathrm{new}})\,
# d_{ji}^{-1}(\mathbf{r}^{\mathrm{old}}) = 
# \sum_{j=1}^N \phi_j(\mathbf{r}_i^{\mathrm{new}})\,
# d_{ji}^{-1}(\mathbf{r}^{\mathrm{old}})
# $$

# What this means is that in order to get the ratio when only the *i*-th
# particle  has been moved, we only need to calculate the dot
# product of the vector $\left(\phi_1(\mathbf{r}_i^\mathrm{new}),\,\dots,\,
# \phi_N(\mathbf{r}_i^\mathrm{new})\right)$ of single particle  wave functions
# evaluated at this new position with the *i*-th column of the inverse
# matrix $\hat{D}^{-1}$ evaluated at the original position. Such
# an operation has a time scaling of $O(N)$. The only extra thing we
# need to do is to maintain the inverse matrix 
# $\hat{D}^{-1}(\mathbf{x}^{\mathrm{old}})$.
