# Harmonic Oscillator

The harmonic oscillator is omnipresent in physics. Although you may think 
of this as being related to springs, it, or an equivalent
mathematical representation, appears in just about any problem where a
mode is sitting near its potential energy minimum. At that point,
$\partial_x V(x)=0$, and the first non-zero term (aside from a
constant) in the potential energy is that of a harmonic oscillator. In
a solid, sound modes (phonons) are built on a picture of coupled
harmonic oscillators, and in relativistic field theory the fundamental
interactions are also built on coupled oscillators positioned
infinitesimally close to one another in space. The phenomena of a
resonance of an oscillator driven at a fixed frequency plays out
repeatedly in atomic, nuclear and high-energy physics, when quantum
mechanically the evolution of a state oscillates according to
$e^{-iEt}$ and exciting discrete quantum states has very similar
mathematics as exciting discrete states of an oscillator.

The potential energy for a single particle as a function of its position $x$ can be written as a Taylor expansion about some point $x_0$

<!-- Equation labels as ordinary links -->
<div id="_auto1"></div>

$$
\begin{equation}
V(x)=V(x_0)+(x-x_0)\left.\partial_xV(x)\right|_{x_0}+\frac{1}{2}(x-x_0)^2\left.\partial_x^2V(x)\right|_{x_0}
+\frac{1}{3!}\left.\partial_x^3V(x)\right|_{x_0}+\cdots
\label{_auto1} \tag{1}
\end{equation}
$$

If the position $x_0$ is at the minimum of the resonance, the first two non-zero terms of the potential are

$$
\begin{eqnarray}
V(x)&\approx& V(x_0)+\frac{1}{2}(x-x_0)^2\left.\partial_x^2V(x)\right|_{x_0},\\
\nonumber
&=&V(x_0)+\frac{1}{2}k(x-x_0)^2,~~~~k\equiv \left.\partial_x^2V(x)\right|_{x_0},\\
\nonumber
F&=&-\partial_xV(x)=-k(x-x_0).
\end{eqnarray}
$$

Put into Newton's 2nd law (assuming $x_0=0$),

$$
\begin{eqnarray}
m\ddot{x}&=&-kx,\\
x&=&A\cos(\omega_0 t-\phi),~~~\omega_0=\sqrt{k/m}.
\end{eqnarray}
$$

Here $A$ and $\phi$ are arbitrary. Equivalently, one could have
written this as $A\cos(\omega_0 t)+B\sin(\omega_0 t)$, or as the real
part of $Ae^{i\omega_0 t}$. In this last case $A$ could be an
arbitrary complex constant. Thus, there are 2 arbitrary constants
(either $A$ and $B$ or $A$ and $\phi$, or the real and imaginary part
of one complex constant. This is the expectation for a second order
differential equation, and also agrees with the physical expectation
that if you know a particle's initial velocity and position you should
be able to define its future motion, and that those two arbitrary
conditions should translate to two arbitrary constants.

A key feature of harmonic motion is that the system repeats itself
after a time $T=1/f$, where $f$ is the frequency, and $\omega=2\pi f$
is the angular frequency. The period of the motion is independent of
the amplitude. However, this independence is only exact when one can
neglect higher terms of the potential, $x^3, x^4\cdots$. Once can
neglect these terms for sufficiently small amplitudes, and for larger
amplitudes the motion is no longer purely sinusoidal, and even though
the motion repeats itself, the time for repeating the motion is no
longer independent of the amplitude.

One can also calculate the velocity and the kinetic energy as a function of time,

$$
\begin{eqnarray}
\dot{x}&=&-\omega_0A\sin(\omega_0 t-\phi),\\
\nonumber
K&=&\frac{1}{2}m\dot{x}^2=\frac{m\omega_0^2A^2}{2}\sin^2(\omega_0t-\phi),\\
\nonumber
&=&\frac{k}{2}A^2\sin^2(\omega_0t-\phi).
\end{eqnarray}
$$

The total energy is then

<!-- Equation labels as ordinary links -->
<div id="_auto2"></div>

$$
\begin{equation}
E=K+V=\frac{1}{2}m\dot{x}^2+\frac{1}{2}kx^2=\frac{1}{2}kA^2.
\label{_auto2} \tag{2}
\end{equation}
$$

The total energy then goes as the square of the amplitude.


A pendulum is an example of a harmonic oscillator. By expanding the
kinetic and potential energies for small angles find the frequency for
a pendulum of length $L$ with all the mass $m$ centered at the end by
writing the eq.s of motion in the form of a harmonic oscillator.

The potential energy and kinetic energies are (for $x$ being the displacement)

$$
\begin{eqnarray*}
V&=&mgL(1-\cos\theta)\approx mgL\frac{x^2}{2L^2},\\
K&=&\frac{1}{2}mL^2\dot{\theta}^2\approx \frac{m}{2}\dot{x}^2.
\end{eqnarray*}
$$

For small $x$ Newton's 2nd law becomes

$$
m\ddot{x}=-\frac{mg}{L}x,
$$

and the spring constant would appear to be $k=mg/L$, which makes the
frequency equal to $\omega_0=\sqrt{g/L}$. Note that the frequency is
independent of the mass.


## Damped Oscillators

We consider only the case where the damping force is proportional to
the velocity. This is counter to dragging friction, where the force is
proportional in strength to the normal force and independent of
velocity, and is also inconsistent with wind resistance, where the
magnitude of the drag force is proportional the square of the
velocity. Rolling resistance does seem to be mainly proportional to
the velocity. However, the main motivation for considering damping
forces proportional to the velocity is that the math is more
friendly. This is because the differential equation is linear,
i.e. each term is of order $x$, $\dot{x}$, $\ddot{x}\cdots$, or even
terms with no mention of $x$, and there are no terms such as $x^2$ or
$x\ddot{x}$. The equations of motion for a spring with damping force
$-b\dot{x}$ are

<!-- Equation labels as ordinary links -->
<div id="_auto3"></div>

$$
\begin{equation}
m\ddot{x}+b\dot{x}+kx=0.
\label{_auto3} \tag{3}
\end{equation}
$$

Just to make the solution a bit less messy, we rewrite this equation as

<!-- Equation labels as ordinary links -->
<div id="eq:dampeddiffyq"></div>

$$
\begin{equation}
\label{eq:dampeddiffyq} \tag{4}
\ddot{x}+2\beta\dot{x}+\omega_0^2x=0,~~~~\beta\equiv b/2m,~\omega_0\equiv\sqrt{k/m}.
\end{equation}
$$

Both $\beta$ and $\omega$ have dimensions of inverse time. To find solutions (see appendix C in the text) you must make an educated guess at the form of the solution. To do this, first realize that the solution will need an arbitrary normalization $A$ because the equation is linear. Secondly, realize that if the form is

<!-- Equation labels as ordinary links -->
<div id="_auto4"></div>

$$
\begin{equation}
x=Ae^{rt}
\label{_auto4} \tag{5}
\end{equation}
$$

that each derivative simply brings out an extra power of $r$. This
means that the $Ae^{rt}$ factors out and one can simply solve for an
equation for $r$. Plugging this form into Eq. ([4](#eq:dampeddiffyq)),

<!-- Equation labels as ordinary links -->
<div id="_auto5"></div>

$$
\begin{equation}
r^2+2\beta r+\omega_0^2=0.
\label{_auto5} \tag{6}
\end{equation}
$$

Because this is a quadratic equation there will be two solutions,

<!-- Equation labels as ordinary links -->
<div id="_auto6"></div>

$$
\begin{equation}
r=-\beta\pm\sqrt{\beta^2-\omega_0^2}.
\label{_auto6} \tag{7}
\end{equation}
$$

We refer to the two solutions as $r_1$ and $r_2$ corresponding to the
$+$ and $-$ roots. As expected, there should be two arbitrary
constants involved in the solution,

<!-- Equation labels as ordinary links -->
<div id="_auto7"></div>

$$
\begin{equation}
x=A_1e^{r_1t}+A_2e^{r_2t},
\label{_auto7} \tag{8}
\end{equation}
$$

where the coefficients $A_1$ and $A_2$ are determined by initial
conditions.

The roots listed above, $\sqrt{\omega_0^2-\beta_0^2}$, will be
imaginary if the damping is small and $\beta<\omega_0$. In that case,
$r$ is complex and the factor $e{rt}$ will have some oscillatory
behavior. If the roots are real, there will only be exponentially
decaying solutions. There are three cases:



### Underdamped: $\beta<\omega_0$

$$
\begin{eqnarray}
x&=&A_1e^{-\beta t}e^{i\omega't}+A_2e^{-\beta t}e^{-i\omega't},~~\omega'\equiv\sqrt{\omega_0^2-\beta^2}\\
\nonumber
&=&(A_1+A_2)e^{-\beta t}\cos\omega't+i(A_1-A_2)e^{-\beta t}\sin\omega't.
\end{eqnarray}
$$

Here we have made use of the identity
$e^{i\omega't}=\cos\omega't+i\sin\omega't$. Because the constants are
arbitrary, and because the real and imaginary parts are both solutions
individually, we can simply consider the real part of the solution
alone:

<!-- Equation labels as ordinary links -->
<div id="eq:homogsolution"></div>

$$
\begin{eqnarray}
\label{eq:homogsolution} \tag{9}
x&=&B_1e^{-\beta t}\cos\omega't+B_2e^{-\beta t}\sin\omega't,\\
\nonumber 
\omega'&\equiv&\sqrt{\omega_0^2-\beta^2}.
\end{eqnarray}
$$

### Critical dampling: $\beta=\omega_0$

In this case the two terms involving $r_1$ and $r_2$ are identical
because $\omega'=0$. Because we need to arbitrary constants, there
needs to be another solution. This is found by simply guessing, or by
taking the limit of $\omega'\rightarrow 0$ from the underdamped
solution. The solution is then

<!-- Equation labels as ordinary links -->
<div id="eq:criticallydamped"></div>

$$
\begin{equation}
\label{eq:criticallydamped} \tag{10}
x=Ae^{-\beta t}+Bte^{-\beta t}.
\end{equation}
$$

The critically damped solution is interesting because the solution
approaches zero quickly, but does not oscillate. For a problem with
zero initial velocity, the solution never crosses zero. This is a good
choice for designing shock absorbers or swinging doors.

### Overdamped: $\beta>\omega_0$

$$
\begin{eqnarray}
x&=&A_1\exp{-(\beta+\sqrt{\beta^2-\omega_0^2})t}+A_2\exp{-(\beta-\sqrt{\beta^2-\omega_0^2})t}
\end{eqnarray}
$$

This solution will also never pass the origin more than once, and then
only if the initial velocity is strong and initially toward zero.




Given $b$, $m$ and $\omega_0$, find $x(t)$ for a particle whose
initial position is $x=0$ and has initial velocity $v_0$ (assuming an
underdamped solution).

The solution is of the form,

$$
\begin{eqnarray*}
x&=&e^{-\beta t}\left[A_1\cos(\omega' t)+A_2\sin\omega't\right],\\
\dot{x}&=&-\beta x+\omega'e^{-\beta t}\left[-A_1\sin\omega't+A_2\cos\omega't\right].\\
\omega'&\equiv&\sqrt{\omega_0^2-\beta^2},~~~\beta\equiv b/2m.
\end{eqnarray*}
$$

From the initial conditions, $A_1=0$ because $x(0)=0$ and $\omega'A_2=v_0$. So

$$
x=\frac{v_0}{\omega'}e^{-\beta t}\sin\omega't.
$$

## Our Sliding Block Code
Here we study first the case without additional friction term and scale our equation
in terms of a dimensionless time $\tau$.

Let us remind ourselves about the differential equation we want to solve (the general case with damping due to friction)

$$
m\frac{d^2x}{dt^2} + b\frac{dx}{dt}+kx(t) =0.
$$

We divide by $m$ and introduce $\omega_0^2=\sqrt{k/m}$ and obtain

$$
\frac{d^2x}{dt^2} + \frac{b}{m}\frac{dx}{dt}+\omega_0^2x(t) =0.
$$

Thereafter we introduce a dimensionless time $\tau = t\omega_0$ (check
that the dimensionality is correct) and rewrite our equation as

$$
\frac{d^2x}{d\tau^2} + \frac{b}{m\omega_0}\frac{dx}{d\tau}+x(\tau) =0,
$$

which gives us

$$
\frac{d^2x}{d\tau^2} + \frac{b}{m\omega_0}\frac{dx}{d\tau}+x(\tau) =0.
$$

We then define $\gamma = b/(2m\omega_0)$ and rewrite our equations as

$$
\frac{d^2x}{d\tau^2} + 2\gamma\frac{dx}{d\tau}+x(\tau) =0.
$$

This is the equation we will code below. The first version employs the Euler-Cromer method.

%matplotlib inline

# Common imports
import numpy as np
import pandas as pd
from math import *
import matplotlib.pyplot as plt
import os

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png')


from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

DeltaT = 0.001
#set up arrays 
tfinal = 20 # in years
n = ceil(tfinal/DeltaT)
# set up arrays for t, v, and x
t = np.zeros(n)
v = np.zeros(n)
x = np.zeros(n)
# Initial conditions as simple one-dimensional arrays of time
x0 =  1.0 
v0 = 0.0
x[0] = x0
v[0] = v0
gamma = 0.0
# Start integrating using Euler-Cromer's method
for i in range(n-1):
    # Set up the acceleration
    # Here you could have defined your own function for this
    a =  -2*gamma*v[i]-x[i]
    # update velocity, time and position
    v[i+1] = v[i] + DeltaT*a
    x[i+1] = x[i] + DeltaT*v[i+1]
    t[i+1] = t[i] + DeltaT
# Plot position as function of time    
fig, ax = plt.subplots()
#ax.set_xlim(0, tfinal)
ax.set_ylabel('x[m]')
ax.set_xlabel('t[s]')
ax.plot(t, x)
fig.tight_layout()
save_fig("BlockEulerCromer")
plt.show()

When setting up the value of $\gamma$ we see that for $\gamma=0$ we get the simple oscillatory motion with no damping.
Choosing $\gamma < 1$ leads to the classical underdamped case with oscillatory motion, but where the motion comes to an end.

Choosing $\gamma =1$ leads to what normally is called critical damping and $\gamma> 1$ leads to critical overdamping.
Try it out and try also to change the initial position and velocity. Setting $\gamma=1$
yields a situation, as discussed above, where the solution approaches quickly zero and does not oscillate. With zero initial velocity it will never cross zero. 


## Sinusoidally Driven Oscillators

Here, we consider the force

<!-- Equation labels as ordinary links -->
<div id="_auto8"></div>

$$
\begin{equation}
F=-kx-b\dot{x}+F_0\cos\omega t,
\label{_auto8} \tag{11}
\end{equation}
$$

which leads to the differential equation

<!-- Equation labels as ordinary links -->
<div id="eq:drivenosc"></div>

$$
\begin{equation}
\label{eq:drivenosc} \tag{12}
\ddot{x}+2\beta\dot{x}+\omega_0^2x=(F_0/m)\cos\omega t.
\end{equation}
$$

Consider a single solution with no arbitrary constants, which we will
call a {\it particular solution}, $x_p(t)$. It should be emphasized
that this is {\bf A} particular solution, because there exists an
infinite number of such solutions because the general solution should
have two arbitrary constants. Now consider solutions to the same
equation without the driving term, which include two arbitrary
constants. These are called either {\it homogenous solutions} or {\it
complementary solutions}, and were given in the previous section,
e.g. Eq. ([9](#eq:homogsolution)) for the underdamped case. The
homogenous solution already incorporates the two arbitrary constants,
so any sum of a homogenous solution and a particular solution will
represent the {\it general solution} of the equation. The general
solution incorporates the two arbitrary constants $A$ and $B$ to
accommodate the two initial conditions. One could have picked a
different particular solution, i.e. the original particular solution
plus any homogenous solution with the arbitrary constants $A_p$ and
$B_p$ chosen at will. When one adds in the homogenous solution, which
has adjustable constants with arbitrary constants $A'$ and $B'$, to
the new particular solution, one can get the same general solution by
simply adjusting the new constants such that $A'+A_p=A$ and
$B'+B_p=B$. Thus, the choice of $A_p$ and $B_p$ are irrelevant, and
when choosing the particular solution it is best to make the simplest
choice possible.

To find a particular solution, one first guesses at the form,

<!-- Equation labels as ordinary links -->
<div id="eq:partform"></div>

$$
\begin{equation}
\label{eq:partform} \tag{13}
x_p(t)=D\cos(\omega t-\delta),
\end{equation}
$$

and rewrite the differential equation as

<!-- Equation labels as ordinary links -->
<div id="_auto9"></div>

$$
\begin{equation}
D\left\{-\omega^2\cos(\omega t-\delta)-2\beta\omega\sin(\omega t-\delta)+\omega_0^2\cos(\omega t-\delta)\right\}=\frac{F_0}{m}\cos(\omega t).
\label{_auto9} \tag{14}
\end{equation}
$$

One can now use angle addition formulas to get

$$
\begin{eqnarray}
D\left\{(-\omega^2\cos\delta+2\beta\omega\sin\delta+\omega_0^2\cos\delta)\cos(\omega t)\right.&&\\
\nonumber
\left.+(-\omega^2\sin\delta-2\beta\omega\cos\delta+\omega_0^2\sin\delta)\sin(\omega t)\right\}
&=&\frac{F_0}{m}\cos(\omega t).
\end{eqnarray}
$$

Both the $\cos$ and $\sin$ terms need to equate if the expression is to hold at all times. Thus, this becomes two equations

$$
\begin{eqnarray}
D\left\{-\omega^2\cos\delta+2\beta\omega\sin\delta+\omega_0^2\cos\delta\right\}&=&\frac{F_0}{m}\\
\nonumber
-\omega^2\sin\delta-2\beta\omega\cos\delta+\omega_0^2\sin\delta&=&0.
\end{eqnarray}
$$

After dividing by $\cos\delta$, the lower expression leads to

<!-- Equation labels as ordinary links -->
<div id="_auto10"></div>

$$
\begin{equation}
\tan\delta=\frac{2\beta\omega}{\omega_0^2-\omega^2}.
\label{_auto10} \tag{15}
\end{equation}
$$

Using the identities $\tan^2+1=\csc^2$ and $\sin^2+\cos^2=1$, one can also express $\sin\delta$ and $\cos\delta$,

$$
\begin{eqnarray}
\sin\delta&=&\frac{2\beta\omega}{\sqrt{(\omega_0^2-\omega^2)^2+4\omega^2\beta^2}},\\
\nonumber
\cos\delta&=&\frac{(\omega_0^2-\omega^2)}{\sqrt{(\omega_0^2-\omega^2)^2+4\omega^2\beta^2}}
\end{eqnarray}
$$

Inserting the expressions for $\cos\delta$ and $\sin\delta$ into the expression for $D$,

<!-- Equation labels as ordinary links -->
<div id="eq:Ddrive"></div>

$$
\begin{equation}
\label{eq:Ddrive} \tag{16}
D=\frac{F_0/m}{\sqrt{(\omega_0^2-\omega^2)^2+4\omega^2\beta^2}}.
\end{equation}
$$

For a given initial condition, e.g. initial displacement and velocity,
one must add the homogenous solution then solve for the two arbitrary
constants. However, because the homogenous solutions decay with time
as $e^{-\beta t}$, the particular solution is all that remains at
large times, and is therefore the steady state solution. Because the
arbitrary constants are all in the homogenous solution, all memory of
the initial conditions are lost at large times, $t>>1/\beta$.

The amplitude of the motion, $D$, is linearly proportional to the
driving force ($F_0/m$), but also depends on the driving frequency
$\omega$. For small $\beta$ the maximum will occur at
$\omega=\omega_0$. This is referred to as a resonance. In the limit
$\beta\rightarrow 0$ the amplitude at resonance approaches infinity.


## Alternative Derivation for Driven Oscillators

Here, we derive the same expressions as in Equations ([13](#eq:partform)) and ([16](#eq:Ddrive)) but express the driving forces as

$$
\begin{eqnarray}
F(t)&=&F_0e^{i\omega t},
\end{eqnarray}
$$

rather than as $F_0\cos\omega t$. The real part of $F$ is the same as before. For the differential equation,

<!-- Equation labels as ordinary links -->
<div id="eq:compdrive"></div>

$$
\begin{eqnarray}
\label{eq:compdrive} \tag{17}
\ddot{x}+2\beta\dot{x}+\omega_0^2x&=&\frac{F_0}{m}e^{i\omega t},
\end{eqnarray}
$$

one can treat $x(t)$ as an imaginary function. Because the operations
$d^2/dt^2$ and $d/dt$ are real and thus do not mix the real and
imaginary parts of $x(t)$, Eq. ([17](#eq:compdrive)) is effectively 2
equations. Because $e^{\omega t}=\cos\omega t+i\sin\omega t$, the real
part of the solution for $x(t)$ gives the solution for a driving force
$F_0\cos\omega t$, and the imaginary part of $x$ corresponds to the
case where the driving force is $F_0\sin\omega t$. It is rather easy
to solve for the complex $x$ in this case, and by taking the real part
of the solution, one finds the answer for the $\cos\omega t$ driving
force.

We assume a simple form for the particular solution

<!-- Equation labels as ordinary links -->
<div id="_auto11"></div>

$$
\begin{equation}
x_p=De^{i\omega t},
\label{_auto11} \tag{18}
\end{equation}
$$

where $D$ is a complex constant.

From Eq. ([17](#eq:compdrive)) one inserts the form for $x_p$ above to get

$$
\begin{eqnarray}
D\left\{-\omega^2+2i\beta\omega+\omega_0^2\right\}e^{i\omega t}=(F_0/m)e^{i\omega t},\\
\nonumber
D=\frac{F_0/m}{(\omega_0^2-\omega^2)+2i\beta\omega}.
\end{eqnarray}
$$

The norm and phase for $D=|D|e^{-i\delta}$ can be read by inspection,

<!-- Equation labels as ordinary links -->
<div id="_auto12"></div>

$$
\begin{equation}
|D|=\frac{F_0/m}{\sqrt{(\omega_0^2-\omega^2)^2+4\beta^2\omega^2}},~~~~\tan\delta=\frac{2\beta\omega}{\omega_0^2-\omega^2}.
\label{_auto12} \tag{19}
\end{equation}
$$

This is the same expression for $\delta$ as before. One then finds $x_p(t)$,

<!-- Equation labels as ordinary links -->
<div id="eq:fastdriven1"></div>

$$
\begin{eqnarray}
\label{eq:fastdriven1} \tag{20}
x_p(t)&=&\Re\frac{(F_0/m)e^{i\omega t-i\delta}}{\sqrt{(\omega_0^2-\omega^2)^2+4\beta^2\omega^2}}\\
\nonumber
&=&\frac{(F_0/m)\cos(\omega t-\delta)}{\sqrt{(\omega_0^2-\omega^2)^2+4\beta^2\omega^2}}.
\end{eqnarray}
$$

This is the same answer as before.
If one wished to solve for the case where $F(t)= F_0\sin\omega t$, the imaginary part of the solution would work

<!-- Equation labels as ordinary links -->
<div id="eq:fastdriven2"></div>

$$
\begin{eqnarray}
\label{eq:fastdriven2} \tag{21}
x_p(t)&=&\Im\frac{(F_0/m)e^{i\omega t-i\delta}}{\sqrt{(\omega_0^2-\omega^2)^2+4\beta^2\omega^2}}\\
\nonumber
&=&\frac{(F_0/m)\sin(\omega t-\delta)}{\sqrt{(\omega_0^2-\omega^2)^2+4\beta^2\omega^2}}.
\end{eqnarray}
$$

Consider the damped and driven harmonic oscillator worked out above. Given $F_0, m,\beta$ and $\omega_0$, solve for the complete solution $x(t)$ for the case where $F=F_0\sin\omega t$ with initial conditions $x(t=0)=0$ and $v(t=0)=0$. Assume the underdamped case.

The general solution including the arbitrary constants includes both the homogenous and particular solutions,

$$
\begin{eqnarray*}
x(t)&=&\frac{F_0}{m}\frac{\sin(\omega t-\delta)}{\sqrt{(\omega_0^2-\omega^2)^2+4\beta^2\omega^2}}
+A\cos\omega't e^{-\beta t}+B\sin\omega't e^{-\beta t}.
\end{eqnarray*}
$$

The quantities $\delta$ and $\omega'$ are given earlier in the
section, $\omega'=\sqrt{\omega_0^2-\beta^2},
\delta=\tan^{-1}(2\beta\omega/(\omega_0^2-\omega^2)$. Here, solving
the problem means finding the arbitrary constants $A$ and
$B$. Satisfying the initial conditions for the initial position and
velocity:

$$
\begin{eqnarray*}
x(t=0)=0&=&-\eta\sin\delta+A,\\
v(t=0)=0&=&\omega\eta\cos\delta-\beta A+\omega'B,\\
\eta&\equiv&\frac{F_0}{m}\frac{1}{\sqrt{(\omega_0^2-\omega^2)^2+4\beta^2\omega^2}}.
\end{eqnarray*}
$$

The problem is now reduced to 2 equations and 2 unknowns, $A$ and $B$. The solution is

$$
\begin{eqnarray}
A&=& \eta\sin\delta ,~~~B=\frac{-\omega\eta\cos\delta+\beta\eta\sin\delta}{\omega'}.
\end{eqnarray}
$$

## Resonance Widths; the $Q$ factor

From the previous two sections, the particular solution for a driving force, $F=F_0\cos\omega t$, is

$$
\begin{eqnarray}
x_p(t)&=&\frac{F_0/m}{\sqrt{(\omega_0^2-\omega^2)^2+4\omega^2\beta^2}}\cos(\omega_t-\delta),\\
\nonumber
\delta&=&\tan^{-1}\left(\frac{2\beta\omega}{\omega_0^2-\omega^2}\right).
\end{eqnarray}
$$

If one fixes the driving frequency $\omega$ and adjusts the
fundamental frequency $\omega_0=\sqrt{k/m}$, the maximum amplitude
occurs when $\omega_0=\omega$ because that is when the term from the
denominator $(\omega_0^2-\omega^2)^2+4\omega^2\beta^2$ is at a
minimum. This is akin to dialing into a radio station. However, if one
fixes $\omega_0$ and adjusts the driving frequency one minimize with
respect to $\omega$, e.g. set

<!-- Equation labels as ordinary links -->
<div id="_auto13"></div>

$$
\begin{equation}
\frac{d}{d\omega}\left[(\omega_0^2-\omega^2)^2+4\omega^2\beta^2\right]=0,
\label{_auto13} \tag{22}
\end{equation}
$$

and one finds that the maximum amplitude occurs when
$\omega=\sqrt{\omega_0^2-2\beta^2}$. If $\beta$ is small relative to
$\omega_0$, one can simply state that the maximum amplitude is

<!-- Equation labels as ordinary links -->
<div id="_auto14"></div>

$$
\begin{equation}
x_{\rm max}\approx\frac{F_0}{2m\beta \omega_0}.
\label{_auto14} \tag{23}
\end{equation}
$$

$$
\begin{eqnarray}
\frac{4\omega^2\beta^2}{(\omega_0^2-\omega^2)^2+4\omega^2\beta^2}=\frac{1}{2}.
\end{eqnarray}
$$

For small damping this occurs when $\omega=\omega_0\pm \beta$, so the $FWHM\approx 2\beta$. For the purposes of tuning to a specific frequency, one wants the width to be as small as possible. The ratio of $\omega_0$ to $FWHM$ is known as the {\it quality} factor, or $Q$ factor,

<!-- Equation labels as ordinary links -->
<div id="_auto15"></div>

$$
\begin{equation}
Q\equiv \frac{\omega_0}{2\beta}.
\label{_auto15} \tag{24}
\end{equation}
$$

## Numerical Studies of Driven Oscillations

Solving the problem of driven oscillations numerically gives us much
more flexibility to study different types of driving forces. We can
reuse our earlier code by simply adding a driving force. If we stay in
the $x$-direction only this can be easily done by adding a term
$F_{\mathrm{ext}}(x,t)$. Note that we have kept it rather general
here, allowing for both a spatial and a temporal dependence.

Before we dive into the code, we need to briefly remind ourselves
about the equations we started with for the case with damping, namely

$$
m\frac{d^2x}{dt^2} + b\frac{dx}{dt}+kx(t) =0,
$$

with no external force applied to the system.

Let us now for simplicty assume that our external force is given by

$$
F_{\mathrm{ext}}(t) = F_0\cos{(\omega t)},
$$

where $F_0$ is a constant (what is its dimension?) and $\omega$ is the frequency of the applied external driving force.
**Small question:** would you expect energy to be conserved now?


Introducing the external force into our lovely differential equation
and dividing by $m$ and introducing $\omega_0^2=\sqrt{k/m}$ we have

$$
\frac{d^2x}{dt^2} + \frac{b}{m}\frac{dx}{dt}+\omega_0^2x(t) =\frac{F_0}{m}\cos{(\omega t)},
$$

Thereafter we introduce a dimensionless time $\tau = t\omega_0$
and a dimensionless frequency $\tilde{\omega}=\omega/\omega_0$. We have then

$$
\frac{d^2x}{d\tau^2} + \frac{b}{m\omega_0}\frac{dx}{d\tau}+x(\tau) =\frac{F_0}{m\omega_0^2}\cos{(\tilde{\omega}\tau)},
$$

Introducing a new amplitude $\tilde{F} =F_0/(m\omega_0^2)$ (check dimensionality again) we have

$$
\frac{d^2x}{d\tau^2} + \frac{b}{m\omega_0}\frac{dx}{d\tau}+x(\tau) =\tilde{F}\cos{(\tilde{\omega}\tau)}.
$$

Our final step, as we did in the case of various types of damping, is
to define $\gamma = b/(2m\omega_0)$ and rewrite our equations as

$$
\frac{d^2x}{d\tau^2} + 2\gamma\frac{dx}{d\tau}+x(\tau) =\tilde{F}\cos{(\tilde{\omega}\tau)}.
$$

This is the equation we will code below using the Euler-Cromer method.

DeltaT = 0.001
#set up arrays 
tfinal = 20 # in years
n = ceil(tfinal/DeltaT)
# set up arrays for t, v, and x
t = np.zeros(n)
v = np.zeros(n)
x = np.zeros(n)
# Initial conditions as one-dimensional arrays of time
x0 =  1.0 
v0 = 0.0
x[0] = x0
v[0] = v0
gamma = 0.2
Omegatilde = 0.5
Ftilde = 1.0
# Start integrating using Euler-Cromer's method
for i in range(n-1):
    # Set up the acceleration
    # Here you could have defined your own function for this
    a =  -2*gamma*v[i]-x[i]+Ftilde*cos(t[i]*Omegatilde)
    # update velocity, time and position
    v[i+1] = v[i] + DeltaT*a
    x[i+1] = x[i] + DeltaT*v[i+1]
    t[i+1] = t[i] + DeltaT
# Plot position as function of time    
fig, ax = plt.subplots()
ax.set_ylabel('x[m]')
ax.set_xlabel('t[s]')
ax.plot(t, x)
fig.tight_layout()
save_fig("ForcedBlockEulerCromer")
plt.show()

In the above example we have focused on the Euler-Cromer method. This
method has a local truncation error which is proportional to $\Delta t^2$
and thereby a global error which is proportional to $\Delta t$.
We can improve this by using the Runge-Kutta family of
methods. The widely popular Runge-Kutta to fourth order or just **RK4**
has indeed a much better truncation error. The RK4 method has a global
error which is proportional to $\Delta t$.

Let us revisit this method and see how we can implement it for the above example.



## Differential Equations, Runge-Kutta methods

Runge-Kutta (RK) methods are based on Taylor expansion formulae, but yield
in general better algorithms for solutions of an ordinary differential equation.
The basic philosophy is that it provides an intermediate step in the computation of $y_{i+1}$.

To see this, consider first the following definitions

<!-- Equation labels as ordinary links -->
<div id="_auto16"></div>

$$
\begin{equation}
\frac{dy}{dt}=f(t,y),  
\label{_auto16} \tag{25}
\end{equation}
$$

and

<!-- Equation labels as ordinary links -->
<div id="_auto17"></div>

$$
\begin{equation}
y(t)=\int f(t,y) dt,  
\label{_auto17} \tag{26}
\end{equation}
$$

and

<!-- Equation labels as ordinary links -->
<div id="_auto18"></div>

$$
\begin{equation}
y_{i+1}=y_i+ \int_{t_i}^{t_{i+1}} f(t,y) dt.
\label{_auto18} \tag{27}
\end{equation}
$$

To demonstrate the philosophy behind RK methods, let us consider
the second-order RK method, RK2.
The first approximation consists in Taylor expanding $f(t,y)$
around the center of the integration interval $t_i$ to $t_{i+1}$,
that is, at $t_i+h/2$, $h$ being the step.
Using the midpoint formula for an integral, 
defining $y(t_i+h/2) = y_{i+1/2}$ and   
$t_i+h/2 = t_{i+1/2}$, we obtain

<!-- Equation labels as ordinary links -->
<div id="_auto19"></div>

$$
\begin{equation}
\int_{t_i}^{t_{i+1}} f(t,y) dt \approx hf(t_{i+1/2},y_{i+1/2}) +O(h^3).
\label{_auto19} \tag{28}
\end{equation}
$$

This means in turn that we have

<!-- Equation labels as ordinary links -->
<div id="_auto20"></div>

$$
\begin{equation}
y_{i+1}=y_i + hf(t_{i+1/2},y_{i+1/2}) +O(h^3).
\label{_auto20} \tag{29}
\end{equation}
$$

However, we do not know the value of   $y_{i+1/2}$. Here comes thus the next approximation, namely, we use Euler's
method to approximate $y_{i+1/2}$. We have then

<!-- Equation labels as ordinary links -->
<div id="_auto21"></div>

$$
\begin{equation}
y_{(i+1/2)}=y_i + \frac{h}{2}\frac{dy}{dt}=y(t_i) + \frac{h}{2}f(t_i,y_i).
\label{_auto21} \tag{30}
\end{equation}
$$

This means that we can define the following algorithm for 
the second-order Runge-Kutta method, RK2.

6
0
 
<
<
<
!
!
M
A
T
H
_
B
L
O
C
K

<!-- Equation labels as ordinary links -->
<div id="_auto23"></div>

$$
\begin{equation}
k_2=hf(t_{i+1/2},y_i+k_1/2),
\label{_auto23} \tag{32}
\end{equation}
$$

with the final value

<!-- Equation labels as ordinary links -->
<div id="_auto24"></div>

$$
\begin{equation} 
y_{i+i}\approx y_i + k_2 +O(h^3). 
\label{_auto24} \tag{33}
\end{equation}
$$

The difference between the previous one-step methods 
is that we now need an intermediate step in our evaluation,
namely $t_i+h/2 = t_{(i+1/2)}$ where we evaluate the derivative $f$. 
This involves more operations, but the gain is a better stability
in the solution.

The fourth-order Runge-Kutta, RK4, has the following algorithm

6
3
 
<
<
<
!
!
M
A
T
H
_
B
L
O
C
K

$$
k_3=hf(t_i+h/2,y_i+k_2/2)\hspace{0.5cm}   k_4=hf(t_i+h,y_i+k_3)
$$

with the final result

$$
y_{i+1}=y_i +\frac{1}{6}\left( k_1 +2k_2+2k_3+k_4\right).
$$

Thus, the algorithm consists in first calculating $k_1$ 
with $t_i$, $y_1$ and $f$ as inputs. Thereafter, we increase the step
size by $h/2$ and calculate $k_2$, then $k_3$ and finally $k_4$. The global error goes as $O(h^4)$.


However, at this stage, if we keep adding different methods in our
main program, the code will quickly become messy and ugly. Before we
proceed thus, we will now introduce functions that enbody the various
methods for solving differential equations. This means that we can
separate out these methods in own functions and files (and later as classes and more
generic functions) and simply call them when needed. Similarly, we
could easily encapsulate various forces or other quantities of
interest in terms of functions. To see this, let us bring up the code
we developed above for the simple sliding block, but now only with the simple forward Euler method. We introduce
two functions, one for the simple Euler method and one for the
force.

Note that here the forward Euler method does not know the specific force function to be called.
It receives just an input the name. We can easily change the force by adding another function.

def ForwardEuler(v,x,t,n,Force):
    for i in range(n-1):
        v[i+1] = v[i] + DeltaT*Force(v[i],x[i],t[i])
        x[i+1] = x[i] + DeltaT*v[i]
        t[i+1] = t[i] + DeltaT

def SpringForce(v,x,t):
#   note here that we have divided by mass and we return the acceleration
    return  -2*gamma*v-x+Ftilde*cos(t*Omegatilde)

It is easy to add a new method like the Euler-Cromer

def ForwardEulerCromer(v,x,t,n,Force):
    for i in range(n-1):
        a = Force(v[i],x[i],t[i])
        v[i+1] = v[i] + DeltaT*a
        x[i+1] = x[i] + DeltaT*v[i+1]
        t[i+1] = t[i] + DeltaT

and the Velocity Verlet method (be careful with time-dependence here, it is not an ideal method for non-conservative forces))

def VelocityVerlet(v,x,t,n,Force):
    for i in range(n-1):
        a = Force(v[i],x[i],t[i])
        x[i+1] = x[i] + DeltaT*v[i]+0.5*a
        anew = Force(v[i],x[i+1],t[i+1])
        v[i+1] = v[i] + 0.5*DeltaT*(a+anew)
        t[i+1] = t[i] + DeltaT

Finally, we can now add the Runge-Kutta2 method via a new function

def RK2(v,x,t,n,Force):
    for i in range(n-1):
# Setting up k1
        k1x = DeltaT*v[i]
        k1v = DeltaT*Force(v[i],x[i],t[i])
# Setting up k2
        vv = v[i]+k1v*0.5
        xx = x[i]+k1x*0.5
        k2x = DeltaT*vv
        k2v = DeltaT*Force(vv,xx,t[i]+DeltaT*0.5)
# Final result
        x[i+1] = x[i]+k2x
        v[i+1] = v[i]+k2v
	t[i+1] = t[i]+DeltaT

Finally, we can now add the Runge-Kutta2 method via a new function

def RK4(v,x,t,n,Force):
    for i in range(n-1):
# Setting up k1
        k1x = DeltaT*v[i]
        k1v = DeltaT*Force(v[i],x[i],t[i])
# Setting up k2
        vv = v[i]+k1v*0.5
        xx = x[i]+k1x*0.5
        k2x = DeltaT*vv
        k2v = DeltaT*Force(vv,xx,t[i]+DeltaT*0.5)
# Setting up k3
        vv = v[i]+k2v*0.5
        xx = x[i]+k2x*0.5
        k3x = DeltaT*vv
        k3v = DeltaT*Force(vv,xx,t[i]+DeltaT*0.5)
# Setting up k4
        vv = v[i]+k3v
        xx = x[i]+k3x
        k4x = DeltaT*vv
        k4v = DeltaT*Force(vv,xx,t[i]+DeltaT)
# Final result
        x[i+1] = x[i]+(k1x+2*k2x+2*k3x+k4x)/6.
        v[i+1] = v[i]+(k1v+2*k2v+2*k3v+k4v)/6.
        t[i+1] = t[i] + DeltaT

The Runge-Kutta family of methods are particularly useful when we have a time-dependent acceleration.
If we have forces which depend only the spatial degrees of freedom (no velocity and/or time-dependence), then energy conserving methods like the Velocity Verlet or the Euler-Cromer method are preferred. As soon as we introduce an explicit time-dependence and/or add dissipitave forces like friction or air resistance, then methods like the family of Runge-Kutta methods are well suited for this. 
The code below uses the Runge-Kutta4 methods.

DeltaT = 0.001
#set up arrays 
tfinal = 20 # in years
n = ceil(tfinal/DeltaT)
# set up arrays for t, v, and x
t = np.zeros(n)
v = np.zeros(n)
x = np.zeros(n)
# Initial conditions (can change to more than one dim)
x0 =  1.0 
v0 = 0.0
x[0] = x0
v[0] = v0
gamma = 0.2
Omegatilde = 0.5
Ftilde = 1.0
# Start integrating using Euler's method
# Note that we define the force function as a SpringForce
RK4(v,x,t,n,SpringForce)

# Plot position as function of time    
fig, ax = plt.subplots()
ax.set_ylabel('x[m]')
ax.set_xlabel('t[s]')
ax.plot(t, x)
fig.tight_layout()
save_fig("ForcedBlockRK4")
plt.show()

## Principle of Superposition and Periodic Forces (Fourier Transforms)

If one has several driving forces, $F(t)=\sum_n F_n(t)$, one can find
the particular solution to each $F_n$, $x_{pn}(t)$, and the particular
solution for the entire driving force is

<!-- Equation labels as ordinary links -->
<div id="_auto25"></div>

$$
\begin{equation}
x_p(t)=\sum_nx_{pn}(t).
\label{_auto25} \tag{34}
\end{equation}
$$

This is known as the principal of superposition. It only applies when
the homogenous equation is linear. If there were an anharmonic term
such as $x^3$ in the homogenous equation, then when one summed various
solutions, $x=(\sum_n x_n)^2$, one would get cross
terms. Superposition is especially useful when $F(t)$ can be written
as a sum of sinusoidal terms, because the solutions for each
sinusoidal (sine or cosine)  term is analytic, as we saw above.

Driving forces are often periodic, even when they are not
sinusoidal. Periodicity implies that for some time $\tau$

$$
\begin{eqnarray}
F(t+\tau)=F(t). 
\end{eqnarray}
$$

One example of a non-sinusoidal periodic force is a square wave. Many
components in electric circuits are non-linear, e.g. diodes, which
makes many wave forms non-sinusoidal even when the circuits are being
driven by purely sinusoidal sources.

The code here shows a typical example of such a square wave generated using the functionality included in the **scipy** Python package. We have used a period of $\tau=0.2$.

import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

# number of points                                                                                       
n = 500
# start and final times                                                                                  
t0 = 0.0
tn = 1.0
# Period                                                                                                 
t = np.linspace(t0, tn, n, endpoint=False)
SqrSignal = np.zeros(n)
SqrSignal = 1.0+signal.square(2*np.pi*5*t)
plt.plot(t, SqrSignal)
plt.ylim(-0.5, 2.5)
plt.show()

For the sinusoidal example studied in the previous subsections the
period is $\tau=2\pi/\omega$. However, higher harmonics can also
satisfy the periodicity requirement. In general, any force that
satisfies the periodicity requirement can be expressed as a sum over
harmonics,

<!-- Equation labels as ordinary links -->
<div id="_auto26"></div>

$$
\begin{equation}
F(t)=\frac{f_0}{2}+\sum_{n>0} f_n\cos(2n\pi t/\tau)+g_n\sin(2n\pi t/\tau).
\label{_auto26} \tag{35}
\end{equation}
$$

From the previous subsection, one can write down the answer for
$x_{pn}(t)$, by substituting $f_n/m$ or $g_n/m$ for $F_0/m$ into Eq.s
([20](#eq:fastdriven1)) or ([21](#eq:fastdriven2)) respectively. By
writing each factor $2n\pi t/\tau$ as $n\omega t$, with $\omega\equiv
2\pi/\tau$,

<!-- Equation labels as ordinary links -->
<div id="eq:fourierdef1"></div>

$$
\begin{equation}
\label{eq:fourierdef1} \tag{36}
F(t)=\frac{f_0}{2}+\sum_{n>0}f_n\cos(n\omega t)+g_n\sin(n\omega t).
\end{equation}
$$

The solutions for $x(t)$ then come from replacing $\omega$ with
$n\omega$ for each term in the particular solution in Equations
([13](#eq:partform)) and ([16](#eq:Ddrive)),

$$
\begin{eqnarray}
x_p(t)&=&\frac{f_0}{2k}+\sum_{n>0} \alpha_n\cos(n\omega t-\delta_n)+\beta_n\sin(n\omega t-\delta_n),\\
\nonumber
\alpha_n&=&\frac{f_n/m}{\sqrt{((n\omega)^2-\omega_0^2)+4\beta^2n^2\omega^2}},\\
\nonumber
\beta_n&=&\frac{g_n/m}{\sqrt{((n\omega)^2-\omega_0^2)+4\beta^2n^2\omega^2}},\\
\nonumber
\delta_n&=&\tan^{-1}\left(\frac{2\beta n\omega}{\omega_0^2-n^2\omega^2}\right).
\end{eqnarray}
$$

Because the forces have been applied for a long time, any non-zero
damping eliminates the homogenous parts of the solution, so one need
only consider the particular solution for each $n$.

The problem will considered solved if one can find expressions for the
coefficients $f_n$ and $g_n$, even though the solutions are expressed
as an infinite sum. The coefficients can be extracted from the
function $F(t)$ by

<!-- Equation labels as ordinary links -->
<div id="eq:fourierdef2"></div>

$$
\begin{eqnarray}
\label{eq:fourierdef2} \tag{37}
f_n&=&\frac{2}{\tau}\int_{-\tau/2}^{\tau/2} dt~F(t)\cos(2n\pi t/\tau),\\
\nonumber
g_n&=&\frac{2}{\tau}\int_{-\tau/2}^{\tau/2} dt~F(t)\sin(2n\pi t/\tau).
\end{eqnarray}
$$

To check the consistency of these expressions and to verify
Eq. ([37](#eq:fourierdef2)), one can insert the expansion of $F(t)$ in
Eq. ([36](#eq:fourierdef1)) into the expression for the coefficients in
Eq. ([37](#eq:fourierdef2)) and see whether

$$
\begin{eqnarray}
f_n&=?&\frac{2}{\tau}\int_{-\tau/2}^{\tau/2} dt~\left\{
\frac{f_0}{2}+\sum_{m>0}f_m\cos(m\omega t)+g_m\sin(m\omega t)
\right\}\cos(n\omega t).
\end{eqnarray}
$$

Immediately, one can throw away all the terms with $g_m$ because they
convolute an even and an odd function. The term with $f_0/2$
disappears because $\cos(n\omega t)$ is equally positive and negative
over the interval and will integrate to zero. For all the terms
$f_m\cos(m\omega t)$ appearing in the sum, one can use angle addition
formulas to see that $\cos(m\omega t)\cos(n\omega
t)=(1/2)(\cos[(m+n)\omega t]+\cos[(m-n)\omega t]$. This will integrate
to zero unless $m=n$. In that case the $m=n$ term gives

<!-- Equation labels as ordinary links -->
<div id="_auto27"></div>

$$
\begin{equation}
\int_{-\tau/2}^{\tau/2}dt~\cos^2(m\omega t)=\frac{\tau}{2},
\label{_auto27} \tag{38}
\end{equation}
$$

and

$$
\begin{eqnarray}
f_n&=?&\frac{2}{\tau}\int_{-\tau/2}^{\tau/2} dt~f_n/2\\
\nonumber
&=&f_n~\checkmark.
\end{eqnarray}
$$

The same method can be used to check for the consistency of $g_n$.


Consider the driving force:

<!-- Equation labels as ordinary links -->
<div id="_auto28"></div>

$$
\begin{equation}
F(t)=At/\tau,~~-\tau/2<t<\tau/2,~~~F(t+\tau)=F(t).
\label{_auto28} \tag{39}
\end{equation}
$$

Find the Fourier coefficients $f_n$ and $g_n$ for all $n$ using Eq. ([37](#eq:fourierdef2)).

Only the odd coefficients enter by symmetry, i.e. $f_n=0$. One can find $g_n$ integrating by parts,

<!-- Equation labels as ordinary links -->
<div id="eq:fouriersolution"></div>

$$
\begin{eqnarray}
\label{eq:fouriersolution} \tag{40}
g_n&=&\frac{2}{\tau}\int_{-\tau/2}^{\tau/2}dt~\sin(n\omega t) \frac{At}{\tau}\\
\nonumber
u&=&t,~dv=\sin(n\omega t)dt,~v=-\cos(n\omega t)/(n\omega),\\
\nonumber
g_n&=&\frac{-2A}{n\omega \tau^2}\int_{-\tau/2}^{\tau/2}dt~\cos(n\omega t)
+\left.2A\frac{-t\cos(n\omega t)}{n\omega\tau^2}\right|_{-\tau/2}^{\tau/2}.
\end{eqnarray}
$$

The first term is zero because $\cos(n\omega t)$ will be equally
positive and negative over the interval. Using the fact that
$\omega\tau=2\pi$,

$$
\begin{eqnarray}
g_n&=&-\frac{2A}{2n\pi}\cos(n\omega\tau/2)\\
\nonumber
&=&-\frac{A}{n\pi}\cos(n\pi)\\
\nonumber
&=&\frac{A}{n\pi}(-1)^{n+1}.
\end{eqnarray}
$$

## Fourier Series

More text will come here, chpater 5.7-5.8 of Taylor are discussed
during the lectures. The code here uses the Fourier series discussed
in chapter 5.7 for a square wave signal. The equations for the
coefficients are are discussed in Taylor section 5.7, see Example
5.4. The code here visualizes the various approximations given by
Fourier series compared with a square wave with period $T=0.2$, witth
$0.1$ and max value $F=2$. We see that when we increase the number of
components in the Fourier series, the Fourier series approximation gets closes and closes to the square wave signal.

import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

# number of points                                                                                       
n = 500
# start and final times                                                                                  
t0 = 0.0
tn = 1.0
# Period                                                                                                 
T =0.2
# Max value of square signal                                                                             
Fmax= 2.0
# Width of signal                                                                                        
Width = 0.1
t = np.linspace(t0, tn, n, endpoint=False)
SqrSignal = np.zeros(n)
FourierSeriesSignal = np.zeros(n)
SqrSignal = 1.0+signal.square(2*np.pi*5*t+np.pi*Width/T)
a0 = Fmax*Width/T
FourierSeriesSignal = a0
Factor = 2.0*Fmax/np.pi
for i in range(1,500):
    FourierSeriesSignal += Factor/(i)*np.sin(np.pi*i*Width/T)*np.cos(i*t*2*np.pi/T)
plt.plot(t, SqrSignal)
plt.plot(t, FourierSeriesSignal)
plt.ylim(-0.5, 2.5)
plt.show()

## Solving differential equations with Fouries series

The material here was discussed during the lecture of February 19 and 21.
It is also covered by Taylor in section 5.8.



## Response to Transient Force

Consider a particle at rest in the bottom of an underdamped harmonic
oscillator, that then feels a sudden impulse, or change in momentum,
$I=F\Delta t$ at $t=0$. This increases the velocity immediately by an
amount $v_0=I/m$ while not changing the position. One can then solve
the trajectory by solving Eq. ([9](#eq:homogsolution)) with initial
conditions $v_0=I/m$ and $x_0=0$. This gives

<!-- Equation labels as ordinary links -->
<div id="_auto29"></div>

$$
\begin{equation}
x(t)=\frac{I}{m\omega'}e^{-\beta t}\sin\omega't, ~~t>0.
\label{_auto29} \tag{41}
\end{equation}
$$

Here, $\omega'=\sqrt{\omega_0^2-\beta^2}$. For an impulse $I_i$ that
occurs at time $t_i$ the trajectory would be

<!-- Equation labels as ordinary links -->
<div id="_auto30"></div>

$$
\begin{equation}
x(t)=\frac{I_i}{m\omega'}e^{-\beta (t-t_i)}\sin[\omega'(t-t_i)] \Theta(t-t_i),
\label{_auto30} \tag{42}
\end{equation}
$$

where $\Theta(t-t_i)$ is a step function, i.e. $\Theta(x)$ is zero for
$x<0$ and unity for $x>0$. If there were several impulses linear
superposition tells us that we can sum over each contribution,

<!-- Equation labels as ordinary links -->
<div id="_auto31"></div>

$$
\begin{equation}
x(t)=\sum_i\frac{I_i}{m\omega'}e^{-\beta(t-t_i)}\sin[\omega'(t-t_i)]\Theta(t-t_i)
\label{_auto31} \tag{43}
\end{equation}
$$

Now one can consider a series of impulses at times separated by
$\Delta t$, where each impulse is given by $F_i\Delta t$. The sum
above now becomes an integral,

<!-- Equation labels as ordinary links -->
<div id="eq:Greeny"></div>

$$
\begin{eqnarray}\label{eq:Greeny} \tag{44}
x(t)&=&\int_{-\infty}^\infty dt'~F(t')\frac{e^{-\beta(t-t')}\sin[\omega'(t-t')]}{m\omega'}\Theta(t-t')\\
\nonumber
&=&\int_{-\infty}^\infty dt'~F(t')G(t-t'),\\
\nonumber
G(\Delta t)&=&\frac{e^{-\beta\Delta t}\sin[\omega' \Delta t]}{m\omega'}\Theta(\Delta t)
\end{eqnarray}
$$

The quantity
$e^{-\beta(t-t')}\sin[\omega'(t-t')]/m\omega'\Theta(t-t')$ is called a
Green's function, $G(t-t')$. It describes the response at $t$ due to a
force applied at a time $t'$, and is a function of $t-t'$. The step
function ensures that the response does not occur before the force is
applied. One should remember that the form for $G$ would change if the
oscillator were either critically- or over-damped.

When performing the integral in Eq. ([44](#eq:Greeny)) one can use
angle addition formulas to factor out the part with the $t'$
dependence in the integrand,

<!-- Equation labels as ordinary links -->
<div id="eq:Greeny2"></div>

$$
\begin{eqnarray}
\label{eq:Greeny2} \tag{45}
x(t)&=&\frac{1}{m\omega'}e^{-\beta t}\left[I_c(t)\sin(\omega't)-I_s(t)\cos(\omega't)\right],\\
\nonumber
I_c(t)&\equiv&\int_{-\infty}^t dt'~F(t')e^{\beta t'}\cos(\omega't'),\\
\nonumber
I_s(t)&\equiv&\int_{-\infty}^t dt'~F(t')e^{\beta t'}\sin(\omega't').
\end{eqnarray}
$$

If the time $t$ is beyond any time at which the force acts,
$F(t'>t)=0$, the coefficients $I_c$ and $I_s$ become independent of
$t$.


Consider an undamped oscillator ($\beta\rightarrow 0$), with
characteristic frequency $\omega_0$ and mass $m$, that is at rest
until it feels a force described by a Gaussian form,

$$
\begin{eqnarray*}
F(t)&=&F_0 \exp\left\{\frac{-t^2}{2\tau^2}\right\}.
\end{eqnarray*}
$$

For large times ($t>>\tau$), where the force has died off, find
$x(t)$.\\ Solve for the coefficients $I_c$ and $I_s$ in
Eq. ([45](#eq:Greeny2)). Because the Gaussian is an even function,
$I_s=0$, and one need only solve for $I_c$,

$$
\begin{eqnarray*}
I_c&=&F_0\int_{-\infty}^\infty dt'~e^{-t^{\prime 2}/(2\tau^2)}\cos(\omega_0 t')\\
&=&\Re F_0 \int_{-\infty}^\infty dt'~e^{-t^{\prime 2}/(2\tau^2)}e^{i\omega_0 t'}\\
&=&\Re F_0 \int_{-\infty}^\infty dt'~e^{-(t'-i\omega_0\tau^2)^2/(2\tau^2)}e^{-\omega_0^2\tau^2/2}\\
&=&F_0\tau \sqrt{2\pi} e^{-\omega_0^2\tau^2/2}.
\end{eqnarray*}
$$

The third step involved completing the square, and the final step used the fact that the integral

$$
\begin{eqnarray*}
\int_{-\infty}^\infty dx~e^{-x^2/2}&=&\sqrt{2\pi}.
\end{eqnarray*}
$$

To see that this integral is true, consider the square of the integral, which you can change to polar coordinates,

$$
\begin{eqnarray*}
I&=&\int_{-\infty}^\infty dx~e^{-x^2/2}\\
I^2&=&\int_{-\infty}^\infty dxdy~e^{-(x^2+y^2)/2}\\
&=&2\pi\int_0^\infty rdr~e^{-r^2/2}\\
&=&2\pi.
\end{eqnarray*}
$$

Finally, the expression for $x$ from Eq. ([45](#eq:Greeny2)) is

$$
\begin{eqnarray*}
x(t>>\tau)&=&\frac{F_0\tau}{m\omega_0} \sqrt{2\pi} e^{-\omega_0^2\tau^2/2}\sin(\omega_0t).
\end{eqnarray*}
$$

## The classical pendulum and scaling the equations

Let us end our discussion of oscillations with another classical case, the pendulum.

The angular equation of motion of the pendulum is given by
Newton's equation and with no external force it reads

<!-- Equation labels as ordinary links -->
<div id="_auto32"></div>

$$
\begin{equation}
  ml\frac{d^2\theta}{dt^2}+mgsin(\theta)=0,
\label{_auto32} \tag{46}
\end{equation}
$$

with an angular velocity and acceleration given by

<!-- Equation labels as ordinary links -->
<div id="_auto33"></div>

$$
\begin{equation}
     v=l\frac{d\theta}{dt},
\label{_auto33} \tag{47}
\end{equation}
$$

and

<!-- Equation labels as ordinary links -->
<div id="_auto34"></div>

$$
\begin{equation}
     a=l\frac{d^2\theta}{dt^2}.
\label{_auto34} \tag{48}
\end{equation}
$$

We do however expect that the motion will gradually come to an end due a viscous drag torque acting on the pendulum. 
In the presence of the drag, the above equation becomes

<!-- Equation labels as ordinary links -->
<div id="eq:pend1"></div>

$$
\begin{equation}
   ml\frac{d^2\theta}{dt^2}+\nu\frac{d\theta}{dt}  +mgsin(\theta)=0, \label{eq:pend1} \tag{49}
\end{equation}
$$

where $\nu$ is now a positive constant parameterizing the viscosity
of the medium in question. In order to maintain the motion against
viscosity, it is necessary to add some external driving force. 
We choose here a periodic driving force. The last equation becomes then

<!-- Equation labels as ordinary links -->
<div id="eq:pend2"></div>

$$
\begin{equation}
   ml\frac{d^2\theta}{dt^2}+\nu\frac{d\theta}{dt}  +mgsin(\theta)=Asin(\omega t), \label{eq:pend2} \tag{50}
\end{equation}
$$

with $A$ and $\omega$ two constants representing the amplitude and 
the angular frequency respectively. The latter is called the driving frequency.



We define

$$
\omega_0=\sqrt{g/l},
$$

the so-called natural frequency and the new dimensionless quantities

$$
\hat{t}=\omega_0t,
$$

with the dimensionless driving frequency

$$
\hat{\omega}=\frac{\omega}{\omega_0},
$$

and introducing the quantity $Q$, called the *quality factor*,

$$
Q=\frac{mg}{\omega_0\nu},
$$

and the dimensionless amplitude

$$
\hat{A}=\frac{A}{mg}
$$

## More on the Pendulum

We have

$$
\frac{d^2\theta}{d\hat{t}^2}+\frac{1}{Q}\frac{d\theta}{d\hat{t}}  
     +sin(\theta)=\hat{A}cos(\hat{\omega}\hat{t}).
$$

This equation can in turn be recast in terms of two coupled first-order differential equations as follows

$$
\frac{d\theta}{d\hat{t}}=\hat{v},
$$

and

$$
\frac{d\hat{v}}{d\hat{t}}=-\frac{\hat{v}}{Q}-sin(\theta)+\hat{A}cos(\hat{\omega}\hat{t}).
$$

These are the equations to be solved.  The factor $Q$ represents the
number of oscillations of the undriven system that must occur before
its energy is significantly reduced due to the viscous drag. The
amplitude $\hat{A}$ is measured in units of the maximum possible
gravitational torque while $\hat{\omega}$ is the angular frequency of
the external torque measured in units of the pendulum's natural
frequency.