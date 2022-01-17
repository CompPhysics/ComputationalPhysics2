#!/usr/bin/env python
# coding: utf-8

# # Hartree-Fock methods
# 
# 
# ## Why Hartree-Fock? Derivation of Hartree-Fock equations in coordinate space
# 
# Hartree-Fock (HF) theory is an algorithm for finding an approximative expression for the ground state of a given Hamiltonian. The basic ingredients are
#   * Define a single-particle basis $\{\psi_{\alpha}\}$ so that

# $$
# \hat{h}^{\mathrm{HF}}\psi_{\alpha} = \varepsilon_{\alpha}\psi_{\alpha}
# $$

# with the Hartree-Fock Hamiltonian defined as

# $$
# \hat{h}^{\mathrm{HF}}=\hat{t}+\hat{u}_{\mathrm{ext}}+\hat{u}^{\mathrm{HF}}
# $$

# * The term  $\hat{u}^{\mathrm{HF}}$ is a single-particle potential to be determined by the HF algorithm.
# 
#   * The HF algorithm means to choose $\hat{u}^{\mathrm{HF}}$ in order to have

# $$
# \langle \hat{H} \rangle = E^{\mathrm{HF}}= \langle \Phi_0 | \hat{H}|\Phi_0 \rangle
# $$

# that is to find a local minimum with a Slater determinant $\Phi_0$ being the ansatz for the ground state. 
#   * The variational principle ensures that $E^{\mathrm{HF}} \ge E_0$, with $E_0$ the exact ground state energy.
# 
# We will show that the Hartree-Fock Hamiltonian $\hat{h}^{\mathrm{HF}}$ equals our definition of the operator $\hat{f}$ discussed in connection with the new definition of the normal-ordered Hamiltonian (see later lectures), that is we have, for a specific matrix element

# $$
# \langle p |\hat{h}^{\mathrm{HF}}| q \rangle =\langle p |\hat{f}| q \rangle=\langle p|\hat{t}+\hat{u}_{\mathrm{ext}}|q \rangle +\sum_{i\le F} \langle pi | \hat{V} | qi\rangle_{AS},
# $$

# meaning that

# $$
# \langle p|\hat{u}^{\mathrm{HF}}|q\rangle = \sum_{i\le F} \langle pi | \hat{V} | qi\rangle_{AS}.
# $$

# The so-called Hartree-Fock potential $\hat{u}^{\mathrm{HF}}$ brings an explicit medium dependence due to the summation over all single-particle states below the Fermi level $F$. It brings also in an explicit dependence on the two-body interaction (in nuclear physics we can also have complicated three- or higher-body forces). The two-body interaction, with its contribution from the other bystanding fermions, creates an effective mean field in which a given fermion moves, in addition to the external potential $\hat{u}_{\mathrm{ext}}$ which confines the motion of the fermion. For systems like nuclei, there is no external confining potential. Nuclei are examples of self-bound systems, where the binding arises due to the intrinsic nature of the strong force. For nuclear systems thus, there would be no external one-body potential in the Hartree-Fock Hamiltonian. 
# 
# ## Variational Calculus and Lagrangian Multipliers
# 
# The calculus of variations involves 
# problems where the quantity to be minimized or maximized is an integral. 
# 
# In the general case we have an integral of the type

# $$
# E[\Phi]= \int_a^b f(\Phi(x),\frac{\partial \Phi}{\partial x},x)dx,
# $$

# where $E$ is the quantity which is sought minimized or maximized.
# The problem is that although $f$ is a function of the variables $\Phi$, $\partial \Phi/\partial x$ and $x$, the exact dependence of
# $\Phi$ on $x$ is not known.  This means again that even though the integral has fixed limits $a$ and $b$, the path of integration is
# not known. In our case the unknown quantities are the single-particle wave functions and we wish to choose an integration path which makes
# the functional $E[\Phi]$ stationary. This means that we want to find minima, or maxima or saddle points. In physics we search normally for minima.
# Our task is therefore to find the minimum of $E[\Phi]$ so that its variation $\delta E$ is zero  subject to specific
# constraints. In our case the constraints appear as the integral which expresses the orthogonality of the  single-particle wave functions.
# The constraints can be treated via the technique of Lagrangian multipliers
# 
# Let us specialize to the expectation value of the energy for one particle in three-dimensions.
# This expectation value reads

# $$
# E=\int dxdydz \psi^*(x,y,z) \hat{H} \psi(x,y,z),
# $$

# with the constraint

# $$
# \int dxdydz \psi^*(x,y,z) \psi(x,y,z)=1,
# $$

# and a Hamiltonian

# $$
# \hat{H}=-\frac{1}{2}\nabla^2+V(x,y,z).
# $$

# We will, for the sake of notational convenience,  skip the variables $x,y,z$ below, and write for example $V(x,y,z)=V$.
# 
# The integral involving the kinetic energy can be written as, with the function $\psi$ vanishing
# strongly for large values of $x,y,z$ (given here by the limits $a$ and $b$),

# $$
# \int_a^b dxdydz \psi^* \left(-\frac{1}{2}\nabla^2\right) \psi dxdydz = \psi^*\nabla\psi|_a^b+\int_a^b dxdydz\frac{1}{2}\nabla\psi^*\nabla\psi.
# $$

# We will drop the limits $a$ and $b$ in the remaining discussion. 
# Inserting this expression into the expectation value for the energy and taking the variational minimum  we obtain

# $$
# \delta E = \delta \left\{\int dxdydz\left( \frac{1}{2}\nabla\psi^*\nabla\psi+V\psi^*\psi\right)\right\} = 0.
# $$

# The constraint appears in integral form as

# $$
# \int dxdydz \psi^* \psi=\mathrm{constant},
# $$

# and multiplying with a Lagrangian multiplier $\lambda$ and taking the variational minimum we obtain the final variational equation

# $$
# \delta \left\{\int dxdydz\left( \frac{1}{2}\nabla\psi^*\nabla\psi+V\psi^*\psi-\lambda\psi^*\psi\right)\right\} = 0.
# $$

# We introduce the function  $f$

# $$
# f =  \frac{1}{2}\nabla\psi^*\nabla\psi+V\psi^*\psi-\lambda\psi^*\psi=
# \frac{1}{2}(\psi^*_x\psi_x+\psi^*_y\psi_y+\psi^*_z\psi_z)+V\psi^*\psi-\lambda\psi^*\psi,
# $$

# where we have skipped the dependence on $x,y,z$ and introduced the shorthand $\psi_x$, $\psi_y$ and $\psi_z$  for the various derivatives.
# 
# For $\psi^*$ the Euler-Lagrange  equations yield

# $$
# \frac{\partial f}{\partial \psi^*}- \frac{\partial }{\partial x}\frac{\partial f}{\partial \psi^*_x}-\frac{\partial }{\partial y}\frac{\partial f}{\partial \psi^*_y}-\frac{\partial }{\partial z}\frac{\partial f}{\partial \psi^*_z}=0,
# $$

# which results in

# $$
# -\frac{1}{2}(\psi_{xx}+\psi_{yy}+\psi_{zz})+V\psi=\lambda \psi.
# $$

# We can then identify the  Lagrangian multiplier as the energy of the system. The last equation is 
# nothing but the standard 
# Schroedinger equation and the variational  approach discussed here provides 
# a powerful method for obtaining approximate solutions of the wave function.
# 
# 
# 
# ## Derivation of Hartree-Fock equations in coordinate space
# 
# Let us denote the ground state energy by $E_0$. According to the
# variational principle we have

# $$
# E_0 \le E[\Phi] = \int \Phi^*\hat{H}\Phi d\mathbf{\tau}
# $$

# where $\Phi$ is a trial function which we assume to be normalized

# $$
# \int \Phi^*\Phi d\mathbf{\tau} = 1,
# $$

# where we have used the shorthand $d\mathbf{\tau}=dx_1dx_2\dots dx_A$.
# 
# 
# 
# 
# In the Hartree-Fock method the trial function is a Slater
# determinant which can be rewritten as

# $$
# \Psi(x_1,x_2,\dots,x_A,\alpha,\beta,\dots,\nu) = \frac{1}{\sqrt{A!}}\sum_{P} (-)^PP\psi_{\alpha}(x_1)
#     \psi_{\beta}(x_2)\dots\psi_{\nu}(x_A)=\sqrt{A!}\hat{A}\Phi_H,
# $$

# where we have introduced the anti-symmetrization operator $\hat{A}$ defined by the 
# summation over all possible permutations *p* of two fermions.
# It is defined as

# $$
# \hat{A} = \frac{1}{A!}\sum_{p} (-)^p\hat{P},
# $$

# with the the Hartree-function given by the simple product of all possible single-particle function

# $$
# \Phi_H(x_1,x_2,\dots,x_A,\alpha,\beta,\dots,\nu) =
#   \psi_{\alpha}(x_1)
#     \psi_{\beta}(x_2)\dots\psi_{\nu}(x_A).
# $$

# Our functional is written as

# $$
# E[\Phi] = \sum_{\mu=1}^A \int \psi_{\mu}^*(x_i)\hat{h}_0(x_i)\psi_{\mu}(x_i) dx_i 
#   + \frac{1}{2}\sum_{\mu=1}^A\sum_{\nu=1}^A
#    \left[ \int \psi_{\mu}^*(x_i)\psi_{\nu}^*(x_j)\hat{v}(r_{ij})\psi_{\mu}(x_i)\psi_{\nu}(x_j)dx_idx_j- \int \psi_{\mu}^*(x_i)\psi_{\nu}^*(x_j)
#  \hat{v}(r_{ij})\psi_{\nu}(x_i)\psi_{\mu}(x_j)dx_idx_j\right]
# $$

# The more compact version reads

# $$
# E[\Phi] 
#   = \sum_{\mu}^A \langle \mu | \hat{h}_0 | \mu\rangle+ \frac{1}{2}\sum_{\mu\nu}^A\left[\langle \mu\nu |\hat{v}|\mu\nu\rangle-\langle \nu\mu |\hat{v}|\mu\nu\rangle\right].
# $$

# Since the interaction is invariant under the interchange of two particles it means for example that we have

# $$
# \langle \mu\nu|\hat{v}|\mu\nu\rangle =  \langle \nu\mu|\hat{v}|\nu\mu\rangle,
# $$

# or in the more general case

# $$
# \langle \mu\nu|\hat{v}|\sigma\tau\rangle =  \langle \nu\mu|\hat{v}|\tau\sigma\rangle.
# $$

# The direct and exchange matrix elements can be  brought together if we define the antisymmetrized matrix element

# $$
# \langle \mu\nu|\hat{v}|\mu\nu\rangle_{AS}= \langle \mu\nu|\hat{v}|\mu\nu\rangle-\langle \mu\nu|\hat{v}|\nu\mu\rangle,
# $$

# or for a general matrix element

# $$
# \langle \mu\nu|\hat{v}|\sigma\tau\rangle_{AS}= \langle \mu\nu|\hat{v}|\sigma\tau\rangle-\langle \mu\nu|\hat{v}|\tau\sigma\rangle.
# $$

# It has the symmetry property

# $$
# \langle \mu\nu|\hat{v}|\sigma\tau\rangle_{AS}= -\langle \mu\nu|\hat{v}|\tau\sigma\rangle_{AS}=-\langle \nu\mu|\hat{v}|\sigma\tau\rangle_{AS}.
# $$

# The antisymmetric matrix element is also hermitian, implying

# $$
# \langle \mu\nu|\hat{v}|\sigma\tau\rangle_{AS}= \langle \sigma\tau|\hat{v}|\mu\nu\rangle_{AS}.
# $$

# With these notations we rewrite the Hartree-Fock functional as

# <!-- Equation labels as ordinary links -->
# <div id="H2Expectation2"></div>
# 
# $$
# \begin{equation}
#   \int \Phi^*\hat{H_I}\Phi d\mathbf{\tau} 
#   = \frac{1}{2}\sum_{\mu=1}^A\sum_{\nu=1}^A \langle \mu\nu|\hat{v}|\mu\nu\rangle_{AS}. \label{H2Expectation2} \tag{1}
# \end{equation}
# $$

# Adding the contribution from the one-body operator $\hat{H}_0$ to
# ([1](#H2Expectation2)) we obtain the energy functional

# <!-- Equation labels as ordinary links -->
# <div id="FunctionalEPhi"></div>
# 
# $$
# \begin{equation}
#   E[\Phi] 
#   = \sum_{\mu=1}^A \langle \mu | h | \mu \rangle +
#   \frac{1}{2}\sum_{{\mu}=1}^A\sum_{{\nu}=1}^A \langle \mu\nu|\hat{v}|\mu\nu\rangle_{AS}. \label{FunctionalEPhi} \tag{2}
# \end{equation}
# $$

# In our coordinate space derivations below we will spell out the Hartree-Fock equations in terms of their integrals.
# 
# 
# 
# 
# If we generalize the Euler-Lagrange equations to more variables 
# and introduce $N^2$ Lagrange multipliers which we denote by 
# $\epsilon_{\mu\nu}$, we can write the variational equation for the functional of $E$

# $$
# \delta E - \sum_{\mu\nu}^A \epsilon_{\mu\nu} \delta
#   \int \psi_{\mu}^* \psi_{\nu} = 0.
# $$

# For the orthogonal wave functions $\psi_{i}$ this reduces to

# $$
# \delta E - \sum_{\mu=1}^A \epsilon_{\mu} \delta
#   \int \psi_{\mu}^* \psi_{\mu} = 0.
# $$

# Variation with respect to the single-particle wave functions $\psi_{\mu}$ yields then

# 3
# 3
#  
# <
# <
# <
# !
# !
# M
# A
# T
# H
# _
# B
# L
# O
# C
# K

# $$
# \sum_{\mu=1}^A \int \psi_{\mu}^*\hat{h_0}(x_i)\delta\psi_{\mu}
#   dx_i 
#   + \frac{1}{2}\sum_{{\mu}=1}^A\sum_{{\nu}=1}^A \left[ \int
#   \psi_{\mu}^*\psi_{\nu}^*\hat{v}(r_{ij})\delta\psi_{\mu}\psi_{\nu} dx_idx_j- \int
#   \psi_{\mu}^*\psi_{\nu}^*\hat{v}(r_{ij})\psi_{\nu}\delta\psi_{\mu}
#   dx_idx_j \right]-  \sum_{{\mu}=1}^A E_{\mu} \int \delta\psi_{\mu}^*
#   \psi_{\mu}dx_i
#   -  \sum_{{\mu}=1}^A E_{\mu} \int \psi_{\mu}^*
#   \delta\psi_{\mu}dx_i = 0.
# $$

# Although the variations $\delta\psi$ and $\delta\psi^*$ are not
# independent, they may in fact be treated as such, so that the 
# terms dependent on either $\delta\psi$ and $\delta\psi^*$ individually 
# may be set equal to zero. To see this, simply 
# replace the arbitrary variation $\delta\psi$ by $i\delta\psi$, so that
# $\delta\psi^*$ is replaced by $-i\delta\psi^*$, and combine the two
# equations. We thus arrive at the Hartree-Fock equations

# <!-- Equation labels as ordinary links -->
# <div id="eq:hartreefockcoordinatespace"></div>
# 
# $$
# \begin{equation}
# \left[ -\frac{1}{2}\nabla_i^2+ \sum_{\nu=1}^A\int \psi_{\nu}^*(x_j)\hat{v}(r_{ij})\psi_{\nu}(x_j)dx_j \right]\psi_{\mu}(x_i) - \left[ \sum_{{\nu}=1}^A \int\psi_{\nu}^*(x_j)\hat{v}(r_{ij})\psi_{\mu}(x_j) dx_j\right] \psi_{\nu}(x_i) = \epsilon_{\mu} \psi_{\mu}(x_i).  \label{eq:hartreefockcoordinatespace} \tag{3}
# \end{equation}
# $$

# Notice that the integration $\int dx_j$ implies an
# integration over the spatial coordinates $\mathbf{r_j}$ and a summation
# over the spin-coordinate of fermion $j$. We note that the factor of $1/2$ in front of the sum involving the two-body interaction, has been removed. This is due to the fact that we need to vary both $\delta\psi_{\mu}^*$ and
# $\delta\psi_{\nu}^*$. Using the symmetry properties of the two-body interaction and interchanging $\mu$ and $\nu$
# as summation indices, we obtain two identical terms. 
# 
# 
# 
# 
# The two first terms in the last equation are the one-body kinetic energy and the
# electron-nucleus potential. The third or *direct* term is the averaged electronic repulsion of the other
# electrons. As written, the
# term includes the *self-interaction* of 
# electrons when $\mu=\nu$. The self-interaction is cancelled in the fourth
# term, or the *exchange* term. The exchange term results from our
# inclusion of the Pauli principle and the assumed determinantal form of
# the wave-function. Equation ([3](#eq:hartreefockcoordinatespace)), in addition to the kinetic energy and the attraction from the atomic nucleus that confines the motion of a single electron,   represents now the motion of a single-particle modified by the two-body interaction. The additional contribution to the Schroedinger equation due to the two-body interaction, represents a mean field set up by all the other bystanding electrons, the latter given by the sum over all single-particle states occupied by $N$ electrons. 
# 
# The Hartree-Fock equation is an example of an integro-differential equation. These equations involve repeated calculations of integrals, in addition to the solution of a set of coupled differential equations. 
# The Hartree-Fock equations can also be rewritten in terms of an eigenvalue problem. The solution of an eigenvalue problem represents often a more practical algorithm and the  solution of  coupled  integro-differential equations.
# This alternative derivation of the Hartree-Fock equations is given below.
# 
# 
# 
# 
# ## Analysis of Hartree-Fock equations in coordinate space
# 
#   A theoretically convenient form of the
# Hartree-Fock equation is to regard the direct and exchange operator
# defined through

# $$
# V_{\mu}^{d}(x_i) = \int \psi_{\mu}^*(x_j) 
#  \hat{v}(r_{ij})\psi_{\mu}(x_j) dx_j
# $$

# and

# $$
# V_{\mu}^{ex}(x_i) g(x_i) 
#   = \left(\int \psi_{\mu}^*(x_j) 
#  \hat{v}(r_{ij})g(x_j) dx_j
#   \right)\psi_{\mu}(x_i),
# $$

# respectively. 
# 
# 
# 
# 
# The function $g(x_i)$ is an arbitrary function,
# and by the substitution $g(x_i) = \psi_{\nu}(x_i)$
# we get

# $$
# V_{\mu}^{ex}(x_i) \psi_{\nu}(x_i) 
#   = \left(\int \psi_{\mu}^*(x_j) 
#  \hat{v}(r_{ij})\psi_{\nu}(x_j)
#   dx_j\right)\psi_{\mu}(x_i).
# $$

# We may then rewrite the Hartree-Fock equations as

# $$
# \hat{h}^{HF}(x_i) \psi_{\nu}(x_i) = \epsilon_{\nu}\psi_{\nu}(x_i),
# $$

# with

# $$
# \hat{h}^{HF}(x_i)= \hat{h}_0(x_i) + \sum_{\mu=1}^AV_{\mu}^{d}(x_i) -
#   \sum_{\mu=1}^AV_{\mu}^{ex}(x_i),
# $$

# and where $\hat{h}_0(i)$ is the one-body part. The latter is normally chosen as a part which yields solutions in closed form. The harmonic oscilltor is a classical problem thereof.
# We normally rewrite the last equation as

# $$
# \hat{h}^{HF}(x_i)= \hat{h}_0(x_i) + \hat{u}^{HF}(x_i).
# $$

# ## Hartree-Fock by varying the coefficients of a wave function expansion
# 
# Another possibility is to expand the single-particle functions in a known basis  and vary the coefficients, 
# that is, the new single-particle wave function is written as a linear expansion
# in terms of a fixed chosen orthogonal basis (for example the well-known harmonic oscillator functions or the hydrogen-like functions etc).
# We define our new Hartree-Fock single-particle basis by performing a unitary transformation 
# on our previous basis (labelled with greek indices) as

# <!-- Equation labels as ordinary links -->
# <div id="eq:newbasis"></div>
# 
# $$
# \begin{equation}
# \psi_p^{HF}  = \sum_{\lambda} C_{p\lambda}\phi_{\lambda}. \label{eq:newbasis} \tag{4}
# \end{equation}
# $$

# In this case we vary the coefficients $C_{p\lambda}$. If the basis has infinitely many solutions, we need
# to truncate the above sum.  We assume that the basis $\phi_{\lambda}$ is orthogonal.
# 
# 
# 
# 
# It is normal to choose a single-particle basis defined as the eigenfunctions
# of parts of the full Hamiltonian. The typical situation consists of the solutions of the one-body part of the Hamiltonian, that is we have

# $$
# \hat{h}_0\phi_{\lambda}=\epsilon_{\lambda}\phi_{\lambda}.
# $$

# The single-particle wave functions $\phi_{\lambda}(\mathbf{r})$, defined by the quantum numbers $\lambda$ and $\mathbf{r}$
# are defined as the overlap

# $$
# \phi_{\lambda}(\mathbf{r})  = \langle \mathbf{r} | \lambda \rangle .
# $$

# In deriving the Hartree-Fock equations, we  will expand the single-particle functions in a known basis  and vary the coefficients, 
# that is, the new single-particle wave function is written as a linear expansion
# in terms of a fixed chosen orthogonal basis (for example the well-known harmonic oscillator functions or the hydrogen-like functions etc).
# 
# We stated that a unitary transformation keeps the orthogonality. To see this consider first a basis of vectors $\mathbf{v}_i$,

# $$
# \mathbf{v}_i = \begin{bmatrix} v_{i1} \\ \dots \\ \dots \\v_{in} \end{bmatrix}
# $$

# We assume that the basis is orthogonal, that is

# $$
# \mathbf{v}_j^T\mathbf{v}_i = \delta_{ij}.
# $$

# An orthogonal or unitary transformation

# $$
# \mathbf{w}_i=\mathbf{U}\mathbf{v}_i,
# $$

# preserves the dot product and orthogonality since

# $$
# \mathbf{w}_j^T\mathbf{w}_i=(\mathbf{U}\mathbf{v}_j)^T\mathbf{U}\mathbf{v}_i=\mathbf{v}_j^T\mathbf{U}^T\mathbf{U}\mathbf{v}_i= \mathbf{v}_j^T\mathbf{v}_i = \delta_{ij}.
# $$

# This means that if the coefficients $C_{p\lambda}$ belong to a unitary or orthogonal trasformation (using the Dirac bra-ket notation)

# $$
# \vert p\rangle  = \sum_{\lambda} C_{p\lambda}\vert\lambda\rangle,
# $$

# orthogonality is preserved, that is $\langle \alpha \vert \beta\rangle = \delta_{\alpha\beta}$
# and $\langle p \vert q\rangle = \delta_{pq}$. 
# 
# This propertry is extremely useful when we build up a basis of many-body Stater determinant based states. 
# 
# **Note also that although a basis $\vert \alpha\rangle$ contains an infinity of states, for practical calculations we have always to make some truncations.** 
# 
# 
# 
# 
# 
# Before we develop the Hartree-Fock equations, there is another very useful property of determinants that we will use both in connection with Hartree-Fock calculations and later shell-model calculations.  
# 
# Consider the following determinant

# $$
# \left| \begin{array}{cc} \alpha_1b_{11}+\alpha_2sb_{12}& a_{12}\\
#                          \alpha_1b_{21}+\alpha_2b_{22}&a_{22}\end{array} \right|=\alpha_1\left|\begin{array}{cc} b_{11}& a_{12}\\
#                          b_{21}&a_{22}\end{array} \right|+\alpha_2\left| \begin{array}{cc} b_{12}& a_{12}\\b_{22}&a_{22}\end{array} \right|
# $$

# We can generalize this to  an $n\times n$ matrix and have

# $$
# \left| \begin{array}{cccccc} a_{11}& a_{12} & \dots & \sum_{k=1}^n c_k b_{1k} &\dots & a_{1n}\\
# a_{21}& a_{22} & \dots & \sum_{k=1}^n c_k b_{2k} &\dots & a_{2n}\\
# \dots & \dots & \dots & \dots & \dots & \dots \\
# \dots & \dots & \dots & \dots & \dots & \dots \\
# a_{n1}& a_{n2} & \dots & \sum_{k=1}^n c_k b_{nk} &\dots & a_{nn}\end{array} \right|=
# \sum_{k=1}^n c_k\left| \begin{array}{cccccc} a_{11}& a_{12} & \dots &  b_{1k} &\dots & a_{1n}\\
# a_{21}& a_{22} & \dots &  b_{2k} &\dots & a_{2n}\\
# \dots & \dots & \dots & \dots & \dots & \dots\\
# \dots & \dots & \dots & \dots & \dots & \dots\\
# a_{n1}& a_{n2} & \dots &  b_{nk} &\dots & a_{nn}\end{array} \right| .
# $$

# This is a property we will use in our Hartree-Fock discussions. 
# 
# 
# 
# 
# We can generalize the previous results, now 
# with all elements $a_{ij}$  being given as functions of 
# linear combinations  of various coefficients $c$ and elements $b_{ij}$,

# $$
# \left| \begin{array}{cccccc} \sum_{k=1}^n b_{1k}c_{k1}& \sum_{k=1}^n b_{1k}c_{k2} & \dots & \sum_{k=1}^n b_{1k}c_{kj}  &\dots & \sum_{k=1}^n b_{1k}c_{kn}\\
# \sum_{k=1}^n b_{2k}c_{k1}& \sum_{k=1}^n b_{2k}c_{k2} & \dots & \sum_{k=1}^n b_{2k}c_{kj} &\dots & \sum_{k=1}^n b_{2k}c_{kn}\\
# \dots & \dots & \dots & \dots & \dots & \dots \\
# \dots & \dots & \dots & \dots & \dots &\dots \\
# \sum_{k=1}^n b_{nk}c_{k1}& \sum_{k=1}^n b_{nk}c_{k2} & \dots & \sum_{k=1}^n b_{nk}c_{kj} &\dots & \sum_{k=1}^n b_{nk}c_{kn}\end{array} \right|=det(\mathbf{C})det(\mathbf{B}),
# $$

# where $det(\mathbf{C})$ and $det(\mathbf{B})$ are the determinants of $n\times n$ matrices
# with elements $c_{ij}$ and $b_{ij}$ respectively.  
# This is a property we will use in our Hartree-Fock discussions. Convince yourself about the correctness of the above expression by setting $n=2$. 
# 
# 
# 
# 
# 
# 
# With our definition of the new basis in terms of an orthogonal basis we have

# $$
# \psi_p(x)  = \sum_{\lambda} C_{p\lambda}\phi_{\lambda}(x).
# $$

# If the coefficients $C_{p\lambda}$ belong to an orthogonal or unitary matrix, the new basis
# is also orthogonal. 
# Our Slater determinant in the new basis $\psi_p(x)$ is written as

# $$
# \frac{1}{\sqrt{A!}}
# \left| \begin{array}{ccccc} \psi_{p}(x_1)& \psi_{p}(x_2)& \dots & \dots & \psi_{p}(x_A)\\
#                             \psi_{q}(x_1)&\psi_{q}(x_2)& \dots & \dots & \psi_{q}(x_A)\\  
#                             \dots & \dots & \dots & \dots & \dots \\
#                             \dots & \dots & \dots & \dots & \dots \\
#                      \psi_{t}(x_1)&\psi_{t}(x_2)& \dots & \dots & \psi_{t}(x_A)\end{array} \right|=\frac{1}{\sqrt{A!}}
# \left| \begin{array}{ccccc} \sum_{\lambda} C_{p\lambda}\phi_{\lambda}(x_1)& \sum_{\lambda} C_{p\lambda}\phi_{\lambda}(x_2)& \dots & \dots & \sum_{\lambda} C_{p\lambda}\phi_{\lambda}(x_A)\\
#                             \sum_{\lambda} C_{q\lambda}\phi_{\lambda}(x_1)&\sum_{\lambda} C_{q\lambda}\phi_{\lambda}(x_2)& \dots & \dots & \sum_{\lambda} C_{q\lambda}\phi_{\lambda}(x_A)\\  
#                             \dots & \dots & \dots & \dots & \dots \\
#                             \dots & \dots & \dots & \dots & \dots \\
#                      \sum_{\lambda} C_{t\lambda}\phi_{\lambda}(x_1)&\sum_{\lambda} C_{t\lambda}\phi_{\lambda}(x_2)& \dots & \dots & \sum_{\lambda} C_{t\lambda}\phi_{\lambda}(x_A)\end{array} \right|,
# $$

# which is nothing but $det(\mathbf{C})det(\Phi)$, with $det(\Phi)$ being the determinant given by the basis functions $\phi_{\lambda}(x)$. 
# 
# 
# 
# In our discussions hereafter we will use our definitions of single-particle states above and below the Fermi ($F$) level given by the labels
# $ijkl\dots \le F$ for so-called single-hole states and $abcd\dots > F$ for so-called particle states.
# For general single-particle states we employ the labels $pqrs\dots$. 
# 
# 
# 
# 
# In Eq. ([2](#FunctionalEPhi)), restated here

# $$
# E[\Phi] 
#   = \sum_{\mu=1}^A \langle \mu | h | \mu \rangle +
#   \frac{1}{2}\sum_{{\mu}=1}^A\sum_{{\nu}=1}^A \langle \mu\nu|\hat{v}|\mu\nu\rangle_{AS},
# $$

# we found the expression for the energy functional in terms of the basis function $\phi_{\lambda}(\mathbf{r})$. We then  varied the above energy functional with respect to the basis functions $|\mu \rangle$. 
# Now we are interested in defining a new basis defined in terms of
# a chosen basis as defined in Eq. ([4](#eq:newbasis)). We can then rewrite the energy functional as

# <!-- Equation labels as ordinary links -->
# <div id="FunctionalEPhi2"></div>
# 
# $$
# \begin{equation}
#   E[\Phi^{HF}] 
#   = \sum_{i=1}^A \langle i | h | i \rangle +
#   \frac{1}{2}\sum_{ij=1}^A\langle ij|\hat{v}|ij\rangle_{AS}, \label{FunctionalEPhi2} \tag{5}
# \end{equation}
# $$

# where $\Phi^{HF}$ is the new Slater determinant defined by the new basis of Eq. ([4](#eq:newbasis)). 
# 
# 
# 
# 
# 
# Using Eq. ([4](#eq:newbasis)) we can rewrite Eq. ([5](#FunctionalEPhi2)) as

# <!-- Equation labels as ordinary links -->
# <div id="FunctionalEPhi3"></div>
# 
# $$
# \begin{equation}
#   E[\Psi] 
#   = \sum_{i=1}^A \sum_{\alpha\beta} C^*_{i\alpha}C_{i\beta}\langle \alpha | h | \beta \rangle +
#   \frac{1}{2}\sum_{ij=1}^A\sum_{{\alpha\beta\gamma\delta}} C^*_{i\alpha}C^*_{j\beta}C_{i\gamma}C_{j\delta}\langle \alpha\beta|\hat{v}|\gamma\delta\rangle_{AS}. \label{FunctionalEPhi3} \tag{6}
# \end{equation}
# $$

# We wish now to minimize the above functional. We introduce again a set of Lagrange multipliers, noting that
# since $\langle i | j \rangle = \delta_{i,j}$ and $\langle \alpha | \beta \rangle = \delta_{\alpha,\beta}$, 
# the coefficients $C_{i\gamma}$ obey the relation

# $$
# \langle i | j \rangle=\delta_{i,j}=\sum_{\alpha\beta} C^*_{i\alpha}C_{i\beta}\langle \alpha | \beta \rangle=
# \sum_{\alpha} C^*_{i\alpha}C_{i\alpha},
# $$

# which allows us to define a functional to be minimized that reads

# <!-- Equation labels as ordinary links -->
# <div id="_auto1"></div>
# 
# $$
# \begin{equation}
#   F[\Phi^{HF}]=E[\Phi^{HF}] - \sum_{i=1}^A\epsilon_i\sum_{\alpha} C^*_{i\alpha}C_{i\alpha}.
# \label{_auto1} \tag{7}
# \end{equation}
# $$

# Minimizing with respect to $C^*_{i\alpha}$, remembering that the equations for $C^*_{i\alpha}$ and $C_{i\alpha}$
# can be written as two  independent equations, we obtain

# $$
# \frac{d}{dC^*_{i\alpha}}\left[  E[\Phi^{HF}] - \sum_{j}\epsilon_j\sum_{\alpha} C^*_{j\alpha}C_{j\alpha}\right]=0,
# $$

# which yields for every single-particle state $i$ and index $\alpha$ (recalling that the coefficients $C_{i\alpha}$ are matrix elements of a unitary (or orthogonal for a real symmetric matrix) matrix)
# the following Hartree-Fock equations

# $$
# \sum_{\beta} C_{i\beta}\langle \alpha | h | \beta \rangle+
# \sum_{j=1}^A\sum_{\beta\gamma\delta} C^*_{j\beta}C_{j\delta}C_{i\gamma}\langle \alpha\beta|\hat{v}|\gamma\delta\rangle_{AS}=\epsilon_i^{HF}C_{i\alpha}.
# $$

# We can rewrite this equation as (changing dummy variables)

# $$
# \sum_{\beta} \left\{\langle \alpha | h | \beta \rangle+
# \sum_{j}^A\sum_{\gamma\delta} C^*_{j\gamma}C_{j\delta}\langle \alpha\gamma|\hat{v}|\beta\delta\rangle_{AS}\right\}C_{i\beta}=\epsilon_i^{HF}C_{i\alpha}.
# $$

# Note that the sums over greek indices run over the number of basis set functions (in principle an infinite number).
# 
# 
# 
# 
# 
# Defining

# $$
# h_{\alpha\beta}^{HF}=\langle \alpha | h | \beta \rangle+
# \sum_{j=1}^A\sum_{\gamma\delta} C^*_{j\gamma}C_{j\delta}\langle \alpha\gamma|\hat{v}|\beta\delta\rangle_{AS},
# $$

# we can rewrite the new equations as

# <!-- Equation labels as ordinary links -->
# <div id="eq:newhf"></div>
# 
# $$
# \begin{equation}
# \sum_{\beta}h_{\alpha\beta}^{HF}C_{i\beta}=\epsilon_i^{HF}C_{i\alpha}. \label{eq:newhf} \tag{8}
# \end{equation}
# $$

# The latter is nothing but a standard eigenvalue problem. Compared with Eq. ([3](#eq:hartreefockcoordinatespace)),
# we see that we do not need to compute any integrals in an iterative procedure for solving the equations.
# It suffices to tabulate the matrix elements $\langle \alpha | h | \beta \rangle$ and $\langle \alpha\gamma|\hat{v}|\beta\delta\rangle_{AS}$ once and for all. Successive iterations require thus only a look-up in tables over one-body and two-body matrix elements. These details will be discussed below when we solve the Hartree-Fock equations numerical. 
# 
# 
# 
# ## Hartree-Fock algorithm
# 
# Our Hartree-Fock matrix  is thus

# $$
# \hat{h}_{\alpha\beta}^{HF}=\langle \alpha | \hat{h}_0 | \beta \rangle+
# \sum_{j=1}^A\sum_{\gamma\delta} C^*_{j\gamma}C_{j\delta}\langle \alpha\gamma|\hat{v}|\beta\delta\rangle_{AS}.
# $$

# The Hartree-Fock equations are solved in an iterative waym starting with a guess for the coefficients $C_{j\gamma}=\delta_{j,\gamma}$ and solving the equations by diagonalization till the new single-particle energies
# $\epsilon_i^{\mathrm{HF}}$ do not change anymore by a prefixed quantity. 
# 
# 
# 
# 
# Normally we assume that the single-particle basis $|\beta\rangle$ forms an eigenbasis for the operator
# $\hat{h}_0$, meaning that the Hartree-Fock matrix becomes

# $$
# \hat{h}_{\alpha\beta}^{HF}=\epsilon_{\alpha}\delta_{\alpha,\beta}+
# \sum_{j=1}^A\sum_{\gamma\delta} C^*_{j\gamma}C_{j\delta}\langle \alpha\gamma|\hat{v}|\beta\delta\rangle_{AS}.
# $$

# The Hartree-Fock eigenvalue problem

# $$
# \sum_{\beta}\hat{h}_{\alpha\beta}^{HF}C_{i\beta}=\epsilon_i^{\mathrm{HF}}C_{i\alpha},
# $$

# can be written out in a more compact form as

# $$
# \hat{h}^{HF}\hat{C}=\epsilon^{\mathrm{HF}}\hat{C}.
# $$

# The Hartree-Fock equations are, in their simplest form, solved in an iterative way, starting with a guess for the
# coefficients $C_{i\alpha}$. We label the coefficients as $C_{i\alpha}^{(n)}$, where the subscript $n$ stands for iteration $n$.
# To set up the algorithm we can proceed as follows:
# 
#  * We start with a guess $C_{i\alpha}^{(0)}=\delta_{i,\alpha}$. Alternatively, we could have used random starting values as long as the vectors are normalized. Another possibility is to give states below the Fermi level a larger weight.
# 
#  * The Hartree-Fock matrix simplifies then to (assuming that the coefficients $C_{i\alpha} $  are real)

# $$
# \hat{h}_{\alpha\beta}^{HF}=\epsilon_{\alpha}\delta_{\alpha,\beta}+
# \sum_{j = 1}^A\sum_{\gamma\delta} C_{j\gamma}^{(0)}C_{j\delta}^{(0)}\langle \alpha\gamma|\hat{v}|\beta\delta\rangle_{AS}.
# $$

# Solving the Hartree-Fock eigenvalue problem yields then new eigenvectors $C_{i\alpha}^{(1)}$ and eigenvalues
# $\epsilon_i^{HF(1)}$. 
#  * With the new eigenvalues we can set up a new Hartree-Fock potential

# $$
# \sum_{j = 1}^A\sum_{\gamma\delta} C_{j\gamma}^{(1)}C_{j\delta}^{(1)}\langle \alpha\gamma|\hat{v}|\beta\delta\rangle_{AS}.
# $$

# The diagonalization with the new Hartree-Fock potential yields new eigenvectors and eigenvalues.
# This process is continued till for example

# $$
# \frac{\sum_{p} |\epsilon_i^{(n)}-\epsilon_i^{(n-1)}|}{m} \le \lambda,
# $$

# where $\lambda$ is a user prefixed quantity ($\lambda \sim 10^{-8}$ or smaller) and $p$ runs over all calculated single-particle
# energies and $m$ is the number of single-particle states.
# 
# 
# 
# ## Analysis of Hartree-Fock equations and Koopman's theorem
# 
# We can rewrite the ground state energy by adding and subtracting $\hat{u}^{HF}(x_i)$

# $$
# E_0^{HF} =\langle \Phi_0 | \hat{H} | \Phi_0\rangle = 
# \sum_{i\le F}^A \langle i | \hat{h}_0 +\hat{u}^{HF}| j\rangle+ \frac{1}{2}\sum_{i\le F}^A\sum_{j \le F}^A\left[\langle ij |\hat{v}|ij \rangle-\langle ij|\hat{v}|ji\rangle\right]-\sum_{i\le F}^A \langle i |\hat{u}^{HF}| i\rangle,
# $$

# which results in

# $$
# E_0^{HF}
#   = \sum_{i\le F}^A \varepsilon_i^{HF} + \frac{1}{2}\sum_{i\le F}^A\sum_{j \le F}^A\left[\langle ij |\hat{v}|ij \rangle-\langle ij|\hat{v}|ji\rangle\right]-\sum_{i\le F}^A \langle i |\hat{u}^{HF}| i\rangle.
# $$

# Our single-particle states $ijk\dots$ are now single-particle states obtained from the solution of the Hartree-Fock equations.
# 
# 
# 
# Using our definition of the Hartree-Fock single-particle energies we obtain then the following expression for the total ground-state energy

# $$
# E_0^{HF}
#   = \sum_{i\le F}^A \varepsilon_i - \frac{1}{2}\sum_{i\le F}^A\sum_{j \le F}^A\left[\langle ij |\hat{v}|ij \rangle-\langle ij|\hat{v}|ji\rangle\right].
# $$

# This form will be used in our discussion of Koopman's theorem.
# 
# 
# 
# In the   atomic physics case we have

# $$
# E[\Phi^{\mathrm{HF}}(N)] 
#   = \sum_{i=1}^H \langle i | \hat{h}_0 | i \rangle +
#   \frac{1}{2}\sum_{ij=1}^N\langle ij|\hat{v}|ij\rangle_{AS},
# $$

# where $\Phi^{\mathrm{HF}}(N)$ is the new Slater determinant defined by the new basis of Eq. ([4](#eq:newbasis))
# for $N$ electrons (same $Z$).  If we assume that the single-particle wave functions in the new basis do not change 
# when we remove one electron or add one electron, we can then define the corresponding energy for the $N-1$ systems as

# $$
# E[\Phi^{\mathrm{HF}}(N-1)] 
#   = \sum_{i=1; i\ne k}^N \langle i | \hat{h}_0 | i \rangle +
#   \frac{1}{2}\sum_{ij=1;i,j\ne k}^N\langle ij|\hat{v}|ij\rangle_{AS},
# $$

# where we have removed a single-particle state $k\le F$, that is a state below the Fermi level.  
# 
# 
# 
# Calculating the difference

# $$
# E[\Phi^{\mathrm{HF}}(N)]-   E[\Phi^{\mathrm{HF}}(N-1)] = \langle k | \hat{h}_0 | k \rangle +
#   \frac{1}{2}\sum_{i=1;i\ne k}^N\langle ik|\hat{v}|ik\rangle_{AS} + \frac{1}{2}\sum_{j=1;j\ne k}^N\langle kj|\hat{v}|kj\rangle_{AS},
# $$

# we obtain

# $$
# E[\Phi^{\mathrm{HF}}(N)]-   E[\Phi^{\mathrm{HF}}(N-1)] = \langle k | \hat{h}_0 | k \rangle +\sum_{j=1}^N\langle kj|\hat{v}|kj\rangle_{AS}
# $$

# which is just our definition of the Hartree-Fock single-particle energy

# $$
# E[\Phi^{\mathrm{HF}}(N)]-   E[\Phi^{\mathrm{HF}}(N-1)] = \epsilon_k^{\mathrm{HF}}
# $$

# Similarly, we can now compute the difference (we label the single-particle states above the Fermi level as $abcd > F$)

# $$
# E[\Phi^{\mathrm{HF}}(N+1)]-   E[\Phi^{\mathrm{HF}}(N)]= \epsilon_a^{\mathrm{HF}}.
# $$

# These two equations can thus be used to the electron affinity or ionization energies, respectively. 
# Koopman's theorem states that for example the ionization energy of a closed-shell system is given by the energy of the highest occupied single-particle state.  If we assume that changing the number of electrons from $N$ to $N+1$ does not change the Hartree-Fock single-particle energies and eigenfunctions, then Koopman's theorem simply states that the ionization energy of an atom is given by the single-particle energy of the last bound state. In a similar way, we can also define the electron affinities. 
# 
# 
# 
# 
# As an example, consider a simple model for atomic sodium, Na. Neutral sodium has eleven electrons, 
# with the weakest bound one being confined the $3s$ single-particle quantum numbers. The energy needed to remove an electron from neutral sodium is rather small, 5.1391 eV, a feature which pertains to all alkali metals.
# Having performed a  Hartree-Fock calculation for neutral sodium would then allows us to compute the
# ionization energy by using the single-particle energy for the $3s$ states, namely $\epsilon_{3s}^{\mathrm{HF}}$. 
# 
# From these considerations, we see that Hartree-Fock theory allows us to make a connection between experimental 
# observables (here ionization and affinity energies) and the underlying interactions between particles.  
# In this sense, we are now linking the dynamics and structure of a many-body system with the laws of motion which govern the system. Our approach is a reductionistic one, meaning that we want to understand the laws of motion 
# in terms of the particles or degrees of freedom which we believe are the fundamental ones. Our Slater determinant, being constructed as the product of various single-particle functions, follows this philosophy.
# 
# 
# 
# 
# With similar arguments as in atomic physics, we can now use Hartree-Fock theory to make a link
# between nuclear forces and separation energies. Changing to nuclear system, we define

# $$
# E[\Phi^{\mathrm{HF}}(A)] 
#   = \sum_{i=1}^A \langle i | \hat{h}_0 | i \rangle +
#   \frac{1}{2}\sum_{ij=1}^A\langle ij|\hat{v}|ij\rangle_{AS},
# $$

# where $\Phi^{\mathrm{HF}}(A)$ is the new Slater determinant defined by the new basis of Eq. ([4](#eq:newbasis))
# for $A$ nucleons, where $A=N+Z$, with $N$ now being the number of neutrons and $Z$ th enumber of protons.  If we assume again that the single-particle wave functions in the new basis do not change from a nucleus with $A$ nucleons to a nucleus with $A-1$  nucleons, we can then define the corresponding energy for the $A-1$ systems as

# $$
# E[\Phi^{\mathrm{HF}}(A-1)] 
#   = \sum_{i=1; i\ne k}^A \langle i | \hat{h}_0 | i \rangle +
#   \frac{1}{2}\sum_{ij=1;i,j\ne k}^A\langle ij|\hat{v}|ij\rangle_{AS},
# $$

# where we have removed a single-particle state $k\le F$, that is a state below the Fermi level.  
# 
# 
# 
# 
# Calculating the difference

# $$
# E[\Phi^{\mathrm{HF}}(A)]-   E[\Phi^{\mathrm{HF}}(A-1)] 
#   = \langle k | \hat{h}_0 | k \rangle +
#   \frac{1}{2}\sum_{i=1;i\ne k}^A\langle ik|\hat{v}|ik\rangle_{AS} + \frac{1}{2}\sum_{j=1;j\ne k}^A\langle kj|\hat{v}|kj\rangle_{AS},
# $$

# which becomes

# $$
# E[\Phi^{\mathrm{HF}}(A)]-   E[\Phi^{\mathrm{HF}}(A-1)] 
#   = \langle k | \hat{h}_0 | k \rangle +\sum_{j=1}^A\langle kj|\hat{v}|kj\rangle_{AS}
# $$

# which is just our definition of the Hartree-Fock single-particle energy

# $$
# E[\Phi^{\mathrm{HF}}(A)]-   E[\Phi^{\mathrm{HF}}(A-1)] 
#   = \epsilon_k^{\mathrm{HF}}
# $$

# Similarly, we can now compute the difference (recall that the single-particle states $abcd > F$)

# $$
# E[\Phi^{\mathrm{HF}}(A+1)]-   E[\Phi^{\mathrm{HF}}(A)]= \epsilon_a^{\mathrm{HF}}.
# $$

# If we then recall that the binding energy differences

# $$
# BE(A)-BE(A-1) \hspace{0.5cm} \mathrm{and} \hspace{0.5cm} BE(A+1)-BE(A),
# $$

# define the separation energies, we see that the Hartree-Fock single-particle energies can be used to
# define separation energies. We have thus our first link between nuclear forces (included in the potential energy term) and an observable quantity defined by differences in binding energies. 
# 
# 
# 
# 
# We have thus the following interpretations (if the single-particle fields do not change)

# $$
# BE(A)-BE(A-1)\approx  E[\Phi^{\mathrm{HF}}(A)]-   E[\Phi^{\mathrm{HF}}(A-1)] 
#   = \epsilon_k^{\mathrm{HF}},
# $$

# and

# $$
# BE(A+1)-BE(A)\approx  E[\Phi^{\mathrm{HF}}(A+1)]-   E[\Phi^{\mathrm{HF}}(A)] =  \epsilon_a^{\mathrm{HF}}.
# $$

# If  we use $^{16}\mbox{O}$ as our closed-shell nucleus, we could then interpret the separation energy

# $$
# BE(^{16}\mathrm{O})-BE(^{15}\mathrm{O})\approx \epsilon_{0p^{\nu}_{1/2}}^{\mathrm{HF}},
# $$

# and

# $$
# BE(^{16}\mathrm{O})-BE(^{15}\mathrm{N})\approx \epsilon_{0p^{\pi}_{1/2}}^{\mathrm{HF}}.
# $$

# Similalry, we could interpret

# $$
# BE(^{17}\mathrm{O})-BE(^{16}\mathrm{O})\approx \epsilon_{0d^{\nu}_{5/2}}^{\mathrm{HF}},
# $$

# and

# $$
# BE(^{17}\mathrm{F})-BE(^{16}\mathrm{O})\approx\epsilon_{0d^{\pi}_{5/2}}^{\mathrm{HF}}.
# $$

# We can continue like this for all $A\pm 1$ nuclei where $A$ is a good closed-shell (or subshell closure)
# nucleus. Examples are $^{22}\mbox{O}$, $^{24}\mbox{O}$, $^{40}\mbox{Ca}$, $^{48}\mbox{Ca}$, $^{52}\mbox{Ca}$, $^{54}\mbox{Ca}$, $^{56}\mbox{Ni}$, 
# $^{68}\mbox{Ni}$, $^{78}\mbox{Ni}$, $^{90}\mbox{Zr}$, $^{88}\mbox{Sr}$, $^{100}\mbox{Sn}$, $^{132}\mbox{Sn}$ and $^{208}\mbox{Pb}$, to mention some possile cases.
# 
# 
# 
# 
# We can thus make our first interpretation of the separation energies in terms of the simplest
# possible many-body theory. 
# If we also recall that the so-called energy gap for neutrons (or protons) is defined as

# $$
# \Delta S_n= 2BE(N,Z)-BE(N-1,Z)-BE(N+1,Z),
# $$

# for neutrons and the corresponding gap for protons

# $$
# \Delta S_p= 2BE(N,Z)-BE(N,Z-1)-BE(N,Z+1),
# $$

# we can define the neutron and proton energy gaps for $^{16}\mbox{O}$ as

# $$
# \Delta S_{\nu}=\epsilon_{0d^{\nu}_{5/2}}^{\mathrm{HF}}-\epsilon_{0p^{\nu}_{1/2}}^{\mathrm{HF}},
# $$

# and

# $$
# \Delta S_{\pi}=\epsilon_{0d^{\pi}_{5/2}}^{\mathrm{HF}}-\epsilon_{0p^{\pi}_{1/2}}^{\mathrm{HF}}.
# $$

# <!-- --- begin exercise --- -->
# 
# ## Exercise 1: Derivation of Hartree-Fock equations
# 
# Consider a Slater determinant built up of single-particle orbitals $\psi_{\lambda}$, 
# with $\lambda = 1,2,\dots,N$.
# 
# The unitary transformation

# $$
# \psi_a  = \sum_{\lambda} C_{a\lambda}\phi_{\lambda},
# $$

# brings us into the new basis.  
# The new basis has quantum numbers $a=1,2,\dots,N$.
# 
# 
# **a)**
# Show that the new basis is orthonormal.
# 
# **b)**
# Show that the new Slater determinant constructed from the new single-particle wave functions can be
# written as the determinant based on the previous basis and the determinant of the matrix $C$.
# 
# **c)**
# Show that the old and the new Slater determinants are equal up to a complex constant with absolute value unity.
# 
# <!-- --- begin hint in exercise --- -->
# 
# **Hint.**
# Use the fact that $C$ is a unitary matrix.
# 
# <!-- --- end hint in exercise --- -->
# 
# 
# 
# 
# <!-- --- end exercise --- -->
# 
# 
# 
# 
# <!-- --- begin exercise --- -->
# 
# ## Exercise 2: Derivation of Hartree-Fock equations
# 
# Consider the  Slater  determinant

# $$
# \Phi_{0}=\frac{1}{\sqrt{n!}}\sum_{p}(-)^{p}P
# \prod_{i=1}^{n}\psi_{\alpha_{i}}(x_{i}).
# $$

# A small variation in this function is given by

# $$
# \delta\Phi_{0}=\frac{1}{\sqrt{n!}}\sum_{p}(-)^{p}P
# \psi_{\alpha_{1}}(x_{1})\psi_{\alpha_{2}}(x_{2})\dots
# \psi_{\alpha_{i-1}}(x_{i-1})(\delta\psi_{\alpha_{i}}(x_{i}))
# \psi_{\alpha_{i+1}}(x_{i+1})\dots\psi_{\alpha_{n}}(x_{n}).
# $$

# **a)**
# Show that

# $$
# \langle \delta\Phi_{0}|\sum_{i=1}^{n}\left\{t(x_{i})+u(x_{i})
# \right\}+\frac{1}{2}
# \sum_{i\neq j=1}^{n}v(x_{i},x_{j})|\Phi_{0}\rangle=\sum_{i=1}^{n}\langle \delta\psi_{\alpha_{i}}|\hat{t}+\hat{u}
# |\phi_{\alpha_{i}}\rangle
# +\sum_{i\neq j=1}^{n}\left\{\langle\delta\psi_{\alpha_{i}}
# \psi_{\alpha_{j}}|\hat{v}|\psi_{\alpha_{i}}\psi_{\alpha_{j}}\rangle-
# \langle\delta\psi_{\alpha_{i}}\psi_{\alpha_{j}}|\hat{v}
# |\psi_{\alpha_{j}}\psi_{\alpha_{i}}\rangle\right\}
# $$

# <!-- --- end exercise --- -->
# 
# 
# 
# 
# <!-- --- begin exercise --- -->
# 
# ## Exercise 3: Developing a  Hartree-Fock program
# 
# Neutron drops are a powerful theoretical laboratory for testing,
# validating and improving nuclear structure models. Indeed, all
# approaches to nuclear structure, from ab initio theory to shell model
# to density functional theory are applicable in such systems. We will,
# therefore, use neutron drops as a test system for setting up a
# Hartree-Fock code.  This program can later be extended to studies of
# the binding energy of nuclei like $^{16}$O or $^{40}$Ca. The
# single-particle energies obtained by solving the Hartree-Fock
# equations can then be directly related to experimental separation
# energies. 
# Since Hartree-Fock theory is the starting point for
# several many-body techniques (density functional theory, random-phase
# approximation, shell-model etc), the aim here is to develop a computer
# program to solve the Hartree-Fock equations in a given single-particle basis,
# here the harmonic oscillator.
# 
# The Hamiltonian for a system of $N$ neutron drops confined in a
# harmonic potential reads

# $$
# \hat{H} = \sum_{i=1}^{N} \frac{\hat{p}_{i}^{2}}{2m}+\sum_{i=1}^{N} \frac{1}{2} m\omega {r}_{i}^{2}+\sum_{i<j} \hat{V}_{ij},
# $$

# with $\hbar^{2}/2m = 20.73$ fm$^{2}$, $mc^{2} = 938.90590$ MeV, and 
# $\hat{V}_{ij}$ is the two-body interaction potential whose 
# matrix elements are precalculated
# and to be read in by you.
# 
# The Hartree-Fock algorithm can be broken down as follows. We recall that  our Hartree-Fock matrix  is

# $$
# \hat{h}_{\alpha\beta}^{HF}=\langle \alpha \vert\hat{h}_0 \vert \beta \rangle+
# \sum_{j=1}^N\sum_{\gamma\delta} C^*_{j\gamma}C_{j\delta}\langle \alpha\gamma|V|\beta\delta\rangle_{AS}.
# $$

# Normally we assume that the single-particle basis $\vert\beta\rangle$
# forms an eigenbasis for the operator $\hat{h}_0$ (this is our case), meaning that the
# Hartree-Fock matrix becomes

# $$
# \hat{h}_{\alpha\beta}^{HF}=\epsilon_{\alpha}\delta_{\alpha,\beta}+
# \sum_{j=1}^N\sum_{\gamma\delta} C^*_{j\gamma}C_{j\delta}\langle \alpha\gamma|V|\beta\delta\rangle_{AS}.
# $$

# The Hartree-Fock eigenvalue problem

# $$
# \sum_{\beta}\hat{h}_{\alpha\beta}^{HF}C_{i\beta}=\epsilon_i^{\mathrm{HF}}C_{i\alpha},
# $$

# can be written out in a more compact form as

# $$
# \hat{h}^{HF}\hat{C}=\epsilon^{\mathrm{HF}}\hat{C}.
# $$

# The equations are often rewritten in terms of a so-called density matrix,
# which is defined as

# <!-- Equation labels as ordinary links -->
# <div id="_auto2"></div>
# 
# $$
# \begin{equation}
# \rho_{\gamma\delta}=\sum_{i=1}^{N}\langle\gamma|i\rangle\langle i|\delta\rangle = \sum_{i=1}^{N}C_{i\gamma}C^*_{i\delta}.
# \label{_auto2} \tag{9}
# \end{equation}
# $$

# It means that we can rewrite the Hartree-Fock Hamiltonian as

# $$
# \hat{h}_{\alpha\beta}^{HF}=\epsilon_{\alpha}\delta_{\alpha,\beta}+
# \sum_{\gamma\delta} \rho_{\gamma\delta}\langle \alpha\gamma|V|\beta\delta\rangle_{AS}.
# $$

# It is convenient to use the density matrix since we can precalculate in every iteration the product of two eigenvector components $C$. 
# 
# 
# Note that $\langle \alpha\vert\hat{h}_0\vert\beta \rangle$ denotes the
# matrix elements of the one-body part of the starting hamiltonian. For
# self-bound nuclei $\langle \alpha\vert\hat{h}_0\vert\beta \rangle$ is the
# kinetic energy, whereas for neutron drops, $\langle \alpha \vert \hat{h}_0 \vert \beta \rangle$ represents the harmonic oscillator hamiltonian since
# the system is confined in a harmonic trap. If we are working in a
# harmonic oscillator basis with the same $\omega$ as the trapping
# potential, then $\langle \alpha\vert\hat{h}_0 \vert \beta \rangle$ is
# diagonal.
# 
# 
# The python
# [program](https://github.com/CompPhysics/ManyBodyMethods/tree/master/doc/src/hfock/Code)
# shows how one can, in a brute force way read in matrix elements in
# $m$-scheme and compute the Hartree-Fock single-particle energies for
# four major shells. The interaction which has been used is the
# so-called N3LO interaction of [Machleidt and
# Entem](http://journals.aps.org/prc/abstract/10.1103/PhysRevC.68.041001)
# using the [Similarity Renormalization
# Group](http://journals.aps.org/prc/abstract/10.1103/PhysRevC.75.061001)
# approach method to renormalize the interaction, using an oscillator
# energy $\hbar\omega=10$ MeV.
# 
# The nucleon-nucleon two-body matrix elements are in $m$-scheme and are fully anti-symmetrized. The Hartree-Fock programs uses the density matrix discussed above in order to compute the Hartree-Fock matrix.
# Here we display the Hartree-Fock part only, assuming that single-particle data and two-body matrix elements have already been read in.

# In[1]:


import numpy as np 
from decimal import Decimal
# expectation value for the one body part, Harmonic oscillator in three dimensions
def onebody(i, n, l):
        homega = 10.0
        return homega*(2*n[i] + l[i] + 1.5)

if __name__ == '__main__':
        
    Nparticles = 16
    """ Read quantum numbers from file """
    index = []
    n = []
    l = []
    j = []
    mj = []
    tz = []
    spOrbitals = 0
    with open("nucleispnumbers.dat", "r") as qnumfile:
                for line in qnumfile:
                        nums = line.split()
                        if len(nums) != 0:
                                index.append(int(nums[0]))
                                n.append(int(nums[1]))
                                l.append(int(nums[2]))
                                j.append(int(nums[3]))
                                mj.append(int(nums[4]))
                                tz.append(int(nums[5]))
                                spOrbitals += 1


    """ Read two-nucleon interaction elements (integrals) from file, brute force 4-dim array """
    nninteraction = np.zeros([spOrbitals, spOrbitals, spOrbitals, spOrbitals])
    with open("nucleitwobody.dat", "r") as infile:
        for line in infile:
                number = line.split()
                a = int(number[0]) - 1
                b = int(number[1]) - 1
                c = int(number[2]) - 1
                d = int(number[3]) - 1
                nninteraction[a][b][c][d] = Decimal(number[4])
        """ Set up single-particle integral """
        singleparticleH = np.zeros(spOrbitals)
        for i in range(spOrbitals):
                singleparticleH[i] = Decimal(onebody(i, n, l))
        
        """ Star HF-iterations, preparing variables and density matrix """

        """ Coefficients for setting up density matrix, assuming only one along the diagonals """
        C = np.eye(spOrbitals) # HF coefficients
        DensityMatrix = np.zeros([spOrbitals,spOrbitals])
        for gamma in range(spOrbitals):
            for delta in range(spOrbitals):
                sum = 0.0
                for i in range(Nparticles):
                    sum += C[gamma][i]*C[delta][i]
                DensityMatrix[gamma][delta] = Decimal(sum)
        maxHFiter = 100
        epsilon =  1.0e-5 
        difference = 1.0
        hf_count = 0
        oldenergies = np.zeros(spOrbitals)
        newenergies = np.zeros(spOrbitals)
        while hf_count < maxHFiter and difference > epsilon:
                print("############### Iteration %i ###############" % hf_count)
                HFmatrix = np.zeros([spOrbitals,spOrbitals])            
                for alpha in range(spOrbitals):
                        for beta in range(spOrbitals):
                            """  If tests for three-dimensional systems, including isospin conservation """
                            if l[alpha] != l[beta] and j[alpha] != j[beta] and mj[alpha] != mj[beta] and tz[alpha] != tz[beta]: continue
                            """  Setting up the Fock matrix using the density matrix and antisymmetrized NN interaction in m-scheme """
                            sumFockTerm = 0.0
                            for gamma in range(spOrbitals):
                                for delta in range(spOrbitals):
                                    if (mj[alpha]+mj[gamma]) != (mj[beta]+mj[delta]) and (tz[alpha]+tz[gamma]) != (tz[beta]+tz[delta]): continue
                                    sumFockTerm += DensityMatrix[gamma][delta]*nninteraction[alpha][gamma][beta][delta]
                            HFmatrix[alpha][beta] = Decimal(sumFockTerm)
                            """  Adding the one-body term, here plain harmonic oscillator """
                            if beta == alpha:   HFmatrix[alpha][alpha] += singleparticleH[alpha]
                spenergies, C = np.linalg.eigh(HFmatrix)
                """ Setting up new density matrix in m-scheme """
                DensityMatrix = np.zeros([spOrbitals,spOrbitals])
                for gamma in range(spOrbitals):
                    for delta in range(spOrbitals):
                        sum = 0.0
                        for i in range(Nparticles):
                            sum += C[gamma][i]*C[delta][i]
                        DensityMatrix[gamma][delta] = Decimal(sum)
                newenergies = spenergies
                """ Brute force computation of difference between previous and new sp HF energies """
                sum =0.0
                for i in range(spOrbitals):
                    sum += (abs(newenergies[i]-oldenergies[i]))/spOrbitals
                difference = sum
                oldenergies = newenergies
                print ("Single-particle energies, ordering may have changed ")
                for i in range(spOrbitals):
                    print('{0:4d}  {1:.4f}'.format(i, Decimal(oldenergies[i])))
                hf_count += 1


# Running the program, one finds that the lowest-lying states for a nucleus like $^{16}\mbox{O}$, we see that the nucleon-nucleon force brings a natural spin-orbit splitting for the $0p$ states (or other states except the $s$-states).
# Since we are using the $m$-scheme for our calculations, we observe that there are several states with the same
# eigenvalues. The number of eigenvalues corresponds to the degeneracy $2j+1$ and is well respected in our calculations, as see from the table here.
# 
# The values of the lowest-lying states are ($\pi$ for protons and $\nu$ for neutrons)
# <table border="1">
# <thead>
# <tr><th align="center">Quantum numbers </th> <th align="center">Energy [MeV]</th> </tr>
# </thead>
# <tbody>
# <tr><td align="center">   $0s_{1/2}^{\pi}$    </td> <td align="center">   -40.4602        </td> </tr>
# <tr><td align="center">   $0s_{1/2}^{\pi}$    </td> <td align="center">   -40.4602        </td> </tr>
# <tr><td align="center">   $0s_{1/2}^{\nu}$    </td> <td align="center">   -40.6426        </td> </tr>
# <tr><td align="center">   $0s_{1/2}^{\nu}$    </td> <td align="center">   -40.6426        </td> </tr>
# <tr><td align="center">   $0p_{1/2}^{\pi}$    </td> <td align="center">   -6.7133         </td> </tr>
# <tr><td align="center">   $0p_{1/2}^{\pi}$    </td> <td align="center">   -6.7133         </td> </tr>
# <tr><td align="center">   $0p_{1/2}^{\nu}$    </td> <td align="center">   -6.8403         </td> </tr>
# <tr><td align="center">   $0p_{1/2}^{\nu}$    </td> <td align="center">   -6.8403         </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\pi}$    </td> <td align="center">   -11.5886        </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\pi}$    </td> <td align="center">   -11.5886        </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\pi}$    </td> <td align="center">   -11.5886        </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\pi}$    </td> <td align="center">   -11.5886        </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\nu}$    </td> <td align="center">   -11.7201        </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\nu}$    </td> <td align="center">   -11.7201        </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\nu}$    </td> <td align="center">   -11.7201        </td> </tr>
# <tr><td align="center">   $0p_{3/2}^{\nu}$    </td> <td align="center">   -11.7201        </td> </tr>
# <tr><td align="center">   $0d_{5/2}^{\pi}$    </td> <td align="center">   18.7589         </td> </tr>
# <tr><td align="center">   $0d_{5/2}^{\nu}$    </td> <td align="center">   18.8082         </td> </tr>
# </tbody>
# </table>
# We can use these results to attempt our first link with experimental data, namely to compute the shell gap or the separation energies. The shell gap for neutrons is given by

# $$
# \Delta S_n= 2BE(N,Z)-BE(N-1,Z)-BE(N+1,Z).
# $$

# For $^{16}\mbox{O}$  we have an experimental value for the  shell gap of $11.51$ MeV for neutrons, while our Hartree-Fock calculations result in $25.65$ MeV. This means that correlations beyond a simple Hartree-Fock calculation with a two-body force play an important role in nuclear physics.
# The splitting between the $0p_{3/2}^{\nu}$ and the $0p_{1/2}^{\nu}$ state is 4.88 MeV, while the experimental value for the gap between the ground state $1/2^{-}$ and the first excited $3/2^{-}$ states is 6.08 MeV. The two-nucleon spin-orbit force plays a central role here. In our discussion of nuclear forces we will see how the spin-orbit force comes into play here.
# 
# <!-- --- end exercise --- -->
# 
# 
# ## Hartree-Fock in second quantization and stability of HF solution
# 
# We wish now to derive the Hartree-Fock equations using our second-quantized formalism and study the stability of the equations. 
# Our ansatz for the ground state of the system is approximated as (this is our representation of a Slater determinant in second quantization)

# $$
# |\Phi_0\rangle = |c\rangle = a^{\dagger}_i a^{\dagger}_j \dots a^{\dagger}_l|0\rangle.
# $$

# We wish to determine $\hat{u}^{HF}$ so that 
# $E_0^{HF}= \langle c|\hat{H}| c\rangle$ becomes a local minimum. 
# 
# In our analysis here we will need Thouless' theorem, which states that
# an arbitrary Slater determinant $|c'\rangle$ which is not orthogonal to a determinant
# $| c\rangle ={\displaystyle\prod_{i=1}^{n}}
# a_{\alpha_{i}}^{\dagger}|0\rangle$, can be written as

# $$
# |c'\rangle=exp\left\{\sum_{a>F}\sum_{i\le F}C_{ai}a_{a}^{\dagger}a_{i}\right\}| c\rangle
# $$

# Let us give a simple proof of Thouless' theorem. The theorem states that we can make a linear combination av particle-hole excitations  with respect to a given reference state $\vert c\rangle$. With this linear combination, we can make a new Slater determinant $\vert c'\rangle $ which is not orthogonal to 
# $\vert c\rangle$, that is

# $$
# \langle c|c'\rangle \ne 0.
# $$

# To show this we need some intermediate steps. The exponential product of two operators  $\exp{\hat{A}}\times\exp{\hat{B}}$ is equal to $\exp{(\hat{A}+\hat{B})}$ only if the two operators commute, that is

# $$
# [\hat{A},\hat{B}] = 0.
# $$

# ## Thouless' theorem
# 
# 
# If the operators do not commute, we need to resort to the [Baker-Campbell-Hauersdorf](http://www.encyclopediaofmath.org/index.php/Campbell%E2%80%93Hausdorff_formula). This relation states that

# $$
# \exp{\hat{C}}=\exp{\hat{A}}\exp{\hat{B}},
# $$

# with

# $$
# \hat{C}=\hat{A}+\hat{B}+\frac{1}{2}[\hat{A},\hat{B}]+\frac{1}{12}[[\hat{A},\hat{B}],\hat{B}]-\frac{1}{12}[[\hat{A},\hat{B}],\hat{A}]+\dots
# $$

# From these relations, we note that 
# in our expression  for $|c'\rangle$ we have commutators of the type

# $$
# [a_{a}^{\dagger}a_{i},a_{b}^{\dagger}a_{j}],
# $$

# and it is easy to convince oneself that these commutators, or higher powers thereof, are all zero. This means that we can write out our new representation of a Slater determinant as

# $$
# |c'\rangle=exp\left\{\sum_{a>F}\sum_{i\le F}C_{ai}a_{a}^{\dagger}a_{i}\right\}| c\rangle=\prod_{i}\left\{1+\sum_{a>F}C_{ai}a_{a}^{\dagger}a_{i}+\left(\sum_{a>F}C_{ai}a_{a}^{\dagger}a_{i}\right)^2+\dots\right\}| c\rangle
# $$

# We note that

# $$
# \prod_{i}\sum_{a>F}C_{ai}a_{a}^{\dagger}a_{i}\sum_{b>F}C_{bi}a_{b}^{\dagger}a_{i}| c\rangle =0,
# $$

# and all higher-order powers of these combinations of creation and annihilation operators disappear 
# due to the fact that $(a_i)^n| c\rangle =0$ when $n > 1$. This allows us to rewrite the expression for $|c'\rangle $ as

# $$
# |c'\rangle=\prod_{i}\left\{1+\sum_{a>F}C_{ai}a_{a}^{\dagger}a_{i}\right\}| c\rangle,
# $$

# which we can rewrite as

# $$
# |c'\rangle=\prod_{i}\left\{1+\sum_{a>F}C_{ai}a_{a}^{\dagger}a_{i}\right\}| a^{\dagger}_{i_1} a^{\dagger}_{i_2} \dots a^{\dagger}_{i_n}|0\rangle.
# $$

# The last equation can be written as

# <!-- Equation labels as ordinary links -->
# <div id="_auto3"></div>
# 
# $$
# \begin{equation}
# |c'\rangle=\prod_{i}\left\{1+\sum_{a>F}C_{ai}a_{a}^{\dagger}a_{i}\right\}| a^{\dagger}_{i_1} a^{\dagger}_{i_2} \dots a^{\dagger}_{i_n}|0\rangle=\left(1+\sum_{a>F}C_{ai_1}a_{a}^{\dagger}a_{i_1}\right)a^{\dagger}_{i_1} 
# \label{_auto3} \tag{10}
# \end{equation}
# $$

# <!-- Equation labels as ordinary links -->
# <div id="_auto4"></div>
# 
# $$
# \begin{equation} 
#  \times\left(1+\sum_{a>F}C_{ai_2}a_{a}^{\dagger}a_{i_2}\right)a^{\dagger}_{i_2} \dots |0\rangle=\prod_{i}\left(a^{\dagger}_{i}+\sum_{a>F}C_{ai}a_{a}^{\dagger}\right)|0\rangle.
# \label{_auto4} \tag{11}
# \end{equation}
# $$

# ## New operators
# 
# 
# If we define a new creation operator

# <!-- Equation labels as ordinary links -->
# <div id="eq:newb"></div>
# 
# $$
# \begin{equation}
# b^{\dagger}_{i}=a^{\dagger}_{i}+\sum_{a>F}C_{ai}a_{a}^{\dagger}, \label{eq:newb} \tag{12}
# \end{equation}
# $$

# we have

# $$
# |c'\rangle=\prod_{i}b^{\dagger}_{i}|0\rangle=\prod_{i}\left(a^{\dagger}_{i}+\sum_{a>F}C_{ai}a_{a}^{\dagger}\right)|0\rangle,
# $$

# meaning that the new representation of the Slater determinant in second quantization, $|c'\rangle$, looks like our previous ones. However, this representation is not general enough since we have a restriction on the sum over single-particle states in Eq. ([12](#eq:newb)). The single-particle states have all to be above the Fermi level.
# The question then is whether we can construct a general representation of a Slater determinant with a creation operator

# $$
# \tilde{b}^{\dagger}_{i}=\sum_{p}f_{ip}a_{p}^{\dagger},
# $$

# where $f_{ip}$ is a matrix element of a unitary matrix which transforms our creation and annihilation operators
# $a^{\dagger}$ and $a$ to $\tilde{b}^{\dagger}$ and $\tilde{b}$. These new operators define a new representation of a Slater determinant as

# $$
# |\tilde{c}\rangle=\prod_{i}\tilde{b}^{\dagger}_{i}|0\rangle.
# $$

# ## Showing that $|\tilde{c}\rangle= |c'\rangle$
# 
# 
# 
# We need to show that $|\tilde{c}\rangle= |c'\rangle$. We need also to assume that the new state
# is not orthogonal to $|c\rangle$, that is $\langle c| \tilde{c}\rangle \ne 0$. From this it follows that

# $$
# \langle c| \tilde{c}\rangle=\langle 0| a_{i_n}\dots a_{i_1}\left(\sum_{p=i_1}^{i_n}f_{i_1p}a_{p}^{\dagger} \right)\left(\sum_{q=i_1}^{i_n}f_{i_2q}a_{q}^{\dagger} \right)\dots \left(\sum_{t=i_1}^{i_n}f_{i_nt}a_{t}^{\dagger} \right)|0\rangle,
# $$

# which is nothing but the determinant $det(f_{ip})$ which we can, using the intermediate normalization condition, 
# normalize to one, that is

# $$
# det(f_{ip})=1,
# $$

# meaning that $f$ has an inverse defined as (since we are dealing with orthogonal, and in our case unitary as well, transformations)

# $$
# \sum_{k} f_{ik}f^{-1}_{kj} = \delta_{ij},
# $$

# and

# $$
# \sum_{j} f^{-1}_{ij}f_{jk} = \delta_{ik}.
# $$

# Using these relations we can then define the linear combination of creation (and annihilation as well) 
# operators as

# $$
# \sum_{i}f^{-1}_{ki}\tilde{b}^{\dagger}_{i}=\sum_{i}f^{-1}_{ki}\sum_{p=i_1}^{\infty}f_{ip}a_{p}^{\dagger}=a_{k}^{\dagger}+\sum_{i}\sum_{p=i_{n+1}}^{\infty}f^{-1}_{ki}f_{ip}a_{p}^{\dagger}.
# $$

# Defining

# $$
# c_{kp}=\sum_{i \le F}f^{-1}_{ki}f_{ip},
# $$

# we can redefine

# $$
# a_{k}^{\dagger}+\sum_{i}\sum_{p=i_{n+1}}^{\infty}f^{-1}_{ki}f_{ip}a_{p}^{\dagger}=a_{k}^{\dagger}+\sum_{p=i_{n+1}}^{\infty}c_{kp}a_{p}^{\dagger}=b_k^{\dagger},
# $$

# our starting point. We have shown that our general representation of a Slater determinant

# $$
# |\tilde{c}\rangle=\prod_{i}\tilde{b}^{\dagger}_{i}|0\rangle=|c'\rangle=\prod_{i}b^{\dagger}_{i}|0\rangle,
# $$

# with

# $$
# b_k^{\dagger}=a_{k}^{\dagger}+\sum_{p=i_{n+1}}^{\infty}c_{kp}a_{p}^{\dagger}.
# $$

# This means that we can actually write an ansatz for the ground state of the system as a linear combination of
# terms which contain the ansatz itself $|c\rangle$ with  an admixture from an infinity of one-particle-one-hole states. The latter has important consequences when we wish to interpret the Hartree-Fock equations and their stability. We can rewrite the new representation as

# $$
# |c'\rangle = |c\rangle+|\delta c\rangle,
# $$

# where $|\delta c\rangle$ can now be interpreted as a small variation. If we approximate this term with 
# contributions from one-particle-one-hole (*1p-1h*) states only, we arrive at

# $$
# |c'\rangle = \left(1+\sum_{ai}\delta C_{ai}a_{a}^{\dagger}a_i\right)|c\rangle.
# $$

# In our derivation of the Hartree-Fock equations we have shown that

# $$
# \langle \delta c| \hat{H} | c\rangle =0,
# $$

# which means that we have to satisfy

# $$
# \langle c|\sum_{ai}\delta C_{ai}\left\{a_{a}^{\dagger}a_i\right\} \hat{H} | c\rangle =0.
# $$

# With this as a background, we are now ready to study the stability of the Hartree-Fock equations.
# 
# 
# 
# ## Hartree-Fock in second quantization and stability of HF solution
# 
# The variational condition for deriving the Hartree-Fock equations guarantees only that the expectation value $\langle c | \hat{H} | c \rangle$ has an extreme value, not necessarily a minimum. To figure out whether the extreme value we have found  is a minimum, we can use second quantization to analyze our results and find a criterion 
# for the above expectation value to a local minimum. We will use Thouless' theorem and show that

# $$
# \frac{\langle c' |\hat{H} | c'\rangle}{\langle c' |c'\rangle} \ge \langle c |\hat{H} | c\rangle= E_0,
# $$

# with

# $$
# {|c'\rangle} = {|c\rangle + |\delta c\rangle}.
# $$

# Using Thouless' theorem we can write out $|c'\rangle$ as

# <!-- Equation labels as ordinary links -->
# <div id="_auto5"></div>
# 
# $$
# \begin{equation}
#  {|c'\rangle}=\exp\left\{\sum_{a > F}\sum_{i \le F}\delta C_{ai}a_{a}^{\dagger}a_{i}\right\}| c\rangle
# \label{_auto5} \tag{13}
# \end{equation}
# $$

# <!-- Equation labels as ordinary links -->
# <div id="_auto6"></div>
# 
# $$
# \begin{equation}  
# =\left\{1+\sum_{a > F}\sum_{i \le F}\delta C_{ai}a_{a}^{\dagger}
# a_{i}+\frac{1}{2!}\sum_{ab > F}\sum_{ij \le F}\delta C_{ai}\delta C_{bj}a_{a}^{\dagger}a_{i}a_{b}^{\dagger}a_{j}+\dots\right\}
# \label{_auto6} \tag{14}
# \end{equation}
# $$

# where the amplitudes $\delta C$ are small.
# 
# 
# The norm of $|c'\rangle$ is given by (using the intermediate normalization condition $\langle c' |c\rangle=1$)

# $$
# \langle c' | c'\rangle = 1+\sum_{a>F}
# \sum_{i\le F}|\delta C_{ai}|^2+O(\delta C_{ai}^3).
# $$

# The expectation value for the energy is now given by (using the Hartree-Fock condition)

# 1
# 4
# 5
#  
# <
# <
# <
# !
# !
# M
# A
# T
# H
# _
# B
# L
# O
# C
# K

# $$
# \frac{1}{2!}\sum_{ab>F}
# \sum_{ij\le F}\delta C_{ai}\delta C_{bj}\langle c |\hat{H}a_{a}^{\dagger}a_{i}a_{b}^{\dagger}a_{j}|c\rangle+\frac{1}{2!}\sum_{ab>F}
# \sum_{ij\le F}\delta C_{ai}^*\delta C_{bj}^*\langle c|a_{j}^{\dagger}a_{b}a_{i}^{\dagger}a_{a}\hat{H}|c\rangle
# +\dots
# $$

# We have already calculated the second term on the right-hand side of the previous equation

# <!-- Equation labels as ordinary links -->
# <div id="_auto7"></div>
# 
# $$
# \begin{equation}
# \langle c | \left(\{a^\dagger_i a_a\} \hat{H} \{a^\dagger_b a_j\} \right) | c\rangle=\sum_{pq} \sum_{ijab}\delta C_{ai}^*\delta C_{bj} \langle p|\hat{h}_0 |q\rangle 
#             \langle c | \left(\{a^{\dagger}_i a_a\}\{a^{\dagger}_pa_q\} 
#              \{a^{\dagger}_b a_j\} \right)| c\rangle
# \label{_auto7} \tag{15}
# \end{equation}
# $$

# <!-- Equation labels as ordinary links -->
# <div id="_auto8"></div>
# 
# $$
# \begin{equation} 
#  +\frac{1}{4} \sum_{pqrs} \sum_{ijab}\delta C_{ai}^*\delta C_{bj} \langle pq| \hat{v}|rs\rangle 
#             \langle c | \left(\{a^\dagger_i a_a\}\{a^{\dagger}_p a^{\dagger}_q a_s  a_r\} \{a^{\dagger}_b a_j\} \right)| c\rangle ,
# \label{_auto8} \tag{16}
# \end{equation}
# $$

# resulting in

# $$
# E_0\sum_{ai}|\delta C_{ai}|^2+\sum_{ai}|\delta C_{ai}|^2(\varepsilon_a-\varepsilon_i)-\sum_{ijab} \langle aj|\hat{v}| bi\rangle \delta C_{ai}^*\delta C_{bj}.
# $$

# $$
# \frac{1}{2!}\langle c |\left(\{a^\dagger_j a_b\} \{a^\dagger_i a_a\} \hat{V}_N  \right) | c\rangle  = 
# \frac{1}{2!}\langle c |\left( \hat{V}_N \{a^\dagger_a a_i\} \{a^\dagger_b a_j\} \right)^{\dagger} | c\rangle
# $$

# which is nothing but

# $$
# \frac{1}{2!}\langle c |  \left( \hat{V}_N \{a^\dagger_a a_i\} \{a^\dagger_b a_j\} \right) | c\rangle^*
# =\frac{1}{2} \sum_{ijab} (\langle ij|\hat{v}|ab\rangle)^*\delta C_{ai}^*\delta C_{bj}^*
# $$

# or

# $$
# \frac{1}{2} \sum_{ijab} (\langle ab|\hat{v}|ij\rangle)\delta C_{ai}^*\delta C_{bj}^*
# $$

# where we have used the relation

# $$
# \langle a |\hat{A} | b\rangle =  (\langle b |\hat{A}^{\dagger} | a\rangle)^*
# $$

# due to the hermiticity of $\hat{H}$ and $\hat{V}$.
# 
# 
# We define two matrix elements

# $$
# A_{ai,bj}=-\langle aj|\hat{v} bi\rangle
# $$

# and

# $$
# B_{ai,bj}=\langle ab|\hat{v}|ij\rangle
# $$

# both being anti-symmetrized.
# 
# 
# 
# With these definitions we write out the energy as

# <!-- Equation labels as ordinary links -->
# <div id="_auto9"></div>
# 
# $$
# \begin{equation}
# \langle c'|H|c'\rangle = \left(1+\sum_{ai}|\delta C_{ai}|^2\right)\langle c |H|c\rangle+\sum_{ai}|\delta C_{ai}|^2(\varepsilon_a^{HF}-\varepsilon_i^{HF})+\sum_{ijab}A_{ai,bj}\delta C_{ai}^*\delta C_{bj}+
# \label{_auto9} \tag{17}
# \end{equation}
# $$

# <!-- Equation labels as ordinary links -->
# <div id="_auto10"></div>
# 
# $$
# \begin{equation} 
# \frac{1}{2} \sum_{ijab} B_{ai,bj}^*\delta C_{ai}\delta C_{bj}+\frac{1}{2} \sum_{ijab} B_{ai,bj}\delta C_{ai}^*\delta C_{bj}^*
# +O(\delta C_{ai}^3),
# \label{_auto10} \tag{18}
# \end{equation}
# $$

# which can be rewritten as

# $$
# \langle c'|H|c'\rangle = \left(1+\sum_{ai}|\delta C_{ai}|^2\right)\langle c |H|c\rangle+\Delta E+O(\delta C_{ai}^3),
# $$

# and skipping higher-order terms we arrived

# $$
# \frac{\langle c' |\hat{H} | c'\rangle}{\langle c' |c'\rangle} =E_0+\frac{\Delta E}{\left(1+\sum_{ai}|\delta C_{ai}|^2\right)}.
# $$

# We have defined

# $$
# \Delta E = \frac{1}{2} \langle \chi | \hat{M}| \chi \rangle
# $$

# with the vectors

# $$
# \chi = \left[ \delta C\hspace{0.2cm} \delta C^*\right]^T
# $$

# and the matrix

# $$
# \hat{M}=\left(\begin{array}{cc} \Delta + A & B \\ B^* & \Delta + A^*\end{array}\right),
# $$

# with $\Delta_{ai,bj} = (\varepsilon_a-\varepsilon_i)\delta_{ab}\delta_{ij}$.
# 
# 
# 
# The condition

# $$
# \Delta E = \frac{1}{2} \langle \chi | \hat{M}| \chi \rangle \ge 0
# $$

# for an arbitrary  vector

# $$
# \chi = \left[ \delta C\hspace{0.2cm} \delta C^*\right]^T
# $$

# means that all eigenvalues of the matrix have to be larger than or equal zero. 
# A necessary (but no sufficient) condition is that the matrix elements (for all $ai$ )

# $$
# (\varepsilon_a-\varepsilon_i)\delta_{ab}\delta_{ij}+A_{ai,bj} \ge 0.
# $$

# This equation can be used as a first test of the stability of the Hartree-Fock equation.
