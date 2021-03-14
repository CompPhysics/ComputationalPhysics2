# Basic Steps of Scientific Investigations

An overarching aim in this course is to give you a deeper
understanding of the scientific method. The problems we study will all
involve cases where we can apply classical mechanics. In our previous
material we already assumed that we had a model for the motion of an
object.  Alternatively we could have data from experiment (like Usain
Bolt's 100m world record run in 2008).  Or we could have performed
ourselves an experiment and we want to understand which forces are at
play and whether these forces can be understood in terms of
fundamental forces.

Our first step consists in identifying the problem. What we sketch
here may include a mix of experiment and theoretical simulations, or
just experiment or only theory.


## Identifying our System

Here we can ask questions like
1. What kind of object is moving

2. What kind of data do we have

3. How do we measure position, velocity, acceleration etc

4. Which initial conditions influence our system

5. Other aspects which allow us to identify the system

## Defining a Model

With our eventual data and observations we would now like to develop a
model for the system. In the end we want obviously to be able to
understand which forces are at play and how they influence our
specific system. That is, can we extract some deeper insights about a
system?

We need then to
1. Find the forces that act on our system

2. Introduce models for the forces

3. Identify the equations which can govern the system (Newton's second law for example)

4. More elements we deem important for defining our model

## Solving the Equations

With the model at hand, we can then solve the equations. In classical mechanics we normally end up  with solving sets of coupled ordinary differential equations or partial differential equations.
1. Using Newton's second law we have equations of the type $\boldsymbol{F}=m\boldsymbol{a}=md\boldsymbol{v}/dt$

2. We need to  define the initial conditions (typically the initial velocity and position as functions of time) and/or initial conditions and boundary conditions

3. The solution of the equations give us then the position, the velocity and other time-dependent quantities which may specify the motion of a given object.

We are not yet done. With our lovely solvers, we need to start thinking.


Now it is time to ask the big questions. What do our results mean? Can we give a simple interpretation in terms of fundamental laws?  What do our results mean? Are they correct?
Thus, typical questions we may ask are
1. Are our results for say $\boldsymbol{r}(t)$ valid?  Do we trust what we did?  Can you validate and verify the correctness of your results?

2. Evaluate the answers and their implications

3. Compare with experimental data if possible. Does our model make sense?

4. and obviously many other questions.

The analysis stage feeds back to the first stage. It may happen that
the data we had were not good enough, there could be large statistical
uncertainties. We may need to collect more data or perhaps we did a
sloppy job in identifying the degrees of freedom.

All these steps are essential elements in a scientific
enquiry. Hopefully, through a mix of numerical simulations, analytical
calculations and experiments we may gain a deeper insight about the
physics of a specific system.

Let us now remind ourselves of Newton's laws, since these are the laws of motion we will study in this course.


## Newton's Laws

When analyzing a physical system we normally start with distinguishing between the object we are studying (we will label this in more general terms as our **system**) and how this system interacts with the environment (which often means everything else!)

In our investigations we will thus analyze a specific physics problem in terms of the system and the environment.
In doing so we need to identify the forces that act on the system and assume that the
forces acting on the system must have a source, an identifiable cause in
the environment.

A force acting on for example a falling object must be related to an interaction with something in the environment.
This also means that we do not consider internal forces. The latter are forces between
one part of the object and another part. In this course we will mainly focus on external forces.

Forces are either contact forces or long-range forces.

Contact forces, as evident from the name, are forces that occur at the contact between
the system and the environment. Well-known long-range forces are the gravitional force and the electromagnetic force.



## Setting up a model for forces acting on an object

In order to set up the forces which act on an object, the following steps may be useful
1. Divide the problem into system and environment.

2. Draw a figure of the object and everything in contact with the object.

3. Draw a closed curve around the system.

4. Find contact points—these are the points where contact forces may act.

5. Give names and symbols to all the contact forces.

6. Identify the long-range forces.

7. Make a drawing of the object. Draw the forces as arrows, vectors, starting from where the force is acting. The direction of the vector(s) indicates the (positive) direction of the force. Try to make the length of the arrow indicate the relative magnitude of the forces.

8. Draw in the axes of the coordinate system. It is often convenient to make one axis parallel to the direction of motion. When you choose the direction of the axis you also choose the positive direction for the axis.

## Newton's Laws, the Second one first


Newton’s second law of motion: The force $\boldsymbol{F}$ on an object of inertial mass $m$
is related to the acceleration a of the object through

$$
\boldsymbol{F} = m\boldsymbol{a},
$$

where $\boldsymbol{a}$ is the acceleration.

Newton’s laws of motion are laws of nature that have been found by experimental
investigations and have been shown to hold up to continued experimental investigations.
Newton’s laws are valid over a wide range of length- and time-scales. We
use Newton’s laws of motion to describe everything from the motion of atoms to the
motion of galaxies.

The second law is a vector equation with the acceleration having the same
direction as the force. The acceleration is proportional to the force via the mass $m$ of the system under study.


Newton’s second law introduces a new property of an object, the so-called 
inertial mass $m$. We determine the inertial mass of an object by measuring the
acceleration for a given applied force.



## Then the First Law


What happens if the net external force on a body is zero? Applying Newton’s second
law, we find:

$$
\boldsymbol{F} = 0 = m\boldsymbol{a},
$$

which gives using the definition of the acceleration

$$
\boldsymbol{a} = \frac{d\boldsymbol{v}}{dt}=0.
$$

The acceleration is zero, which means that the velocity of the object is constant. This
is often referred to as Newton’s first law. An object in a state of uniform motion tends to remain in
that state unless an external force changes its state of motion.
Why do we need a separate law for this? Is it not simply a special case of Newton’s
second law? Yes, Newton’s first law can be deduced from the second law as we have
illustrated. However, the first law is often used for a different purpose: Newton’s
First Law tells us about the limit of applicability of Newton’s Second law. Newton’s
Second law can only be used in reference systems where the First law is obeyed. But
is not the First law always valid? No! The First law is only valid in reference systems
that are not accelerated. If you observe the motion of a ball from an accelerating
car, the ball will appear to accelerate even if there are no forces acting on it. We call
systems that are not accelerating inertial systems, and Newton’s first law is often
called the law of inertia. Newton’s first and second laws of motion are only valid in
inertial systems. 

A system is an inertial system if it is not accelerated. It means that the reference system
must not be accelerating linearly or rotating. Unfortunately, this means that most
systems we know are not really inertial systems. For example, the surface of the
Earth is clearly not an inertial system, because the Earth is rotating. The Earth is also
not an inertial system, because it ismoving in a curved path around the Sun. However,
even if the surface of the Earth is not strictly an inertial system, it may be considered
to be approximately an inertial system for many laboratory-size experiments.


## And finally the Third Law


If there is a force from object A on object B, there is also a force from object B on object A.
This fundamental principle of interactions is called Newton’s third law. We do not
know of any force that do not obey this law: All forces appear in pairs. Newton’s
third law is usually formulated as: For every action there is an equal and opposite
reaction.



## Motion of a Single Object

Here we consider the motion of a single particle moving under
the influence of some set of forces.  We will consider some problems where
the force does not depend on the position. In that case Newton's law
$m\dot{\boldsymbol{v}}=\boldsymbol{F}(\boldsymbol{v})$ is a first-order differential
equation and one solves for $\boldsymbol{v}(t)$, then moves on to integrate
$\boldsymbol{v}$ to get the position. In essentially all of these cases we cna find an analytical solution.



## Air Resistance in One Dimension

Air resistance tends to scale as the square of the velocity. This is
in contrast to many problems chosen for textbooks, where it is linear
in the velocity. The choice of a linear dependence is motivated by
mathematical simplicity (it keeps the differential equation linear)
rather than by physics. One can see that the force should be quadratic
in velocity by considering the momentum imparted on the air
molecules. If an object sweeps through a volume $dV$ of air in time
$dt$, the momentum imparted on the air is

<!-- Equation labels as ordinary links -->
<div id="_auto1"></div>

$$
\begin{equation}
dP=\rho_m dV v,
\label{_auto1} \tag{1}
\end{equation}
$$

where $v$ is the velocity of the object and $\rho_m$ is the mass
density of the air. If the molecules bounce back as opposed to stop
you would double the size of the term. The opposite value of the
momentum is imparted onto the object itself. Geometrically, the
differential volume is

<!-- Equation labels as ordinary links -->
<div id="_auto2"></div>

$$
\begin{equation}
dV=Avdt,
\label{_auto2} \tag{2}
\end{equation}
$$

where $A$ is the cross-sectional area and $vdt$ is the distance the
object moved in time $dt$.


## Resulting Acceleration
Plugging this into the expression above,

<!-- Equation labels as ordinary links -->
<div id="_auto3"></div>

$$
\begin{equation}
\frac{dP}{dt}=-\rho_m A v^2.
\label{_auto3} \tag{3}
\end{equation}
$$

This is the force felt by the particle, and is opposite to its
direction of motion. Now, because air doesn't stop when it hits an
object, but flows around the best it can, the actual force is reduced
by a dimensionless factor $c_W$, called the drag coefficient.

<!-- Equation labels as ordinary links -->
<div id="_auto4"></div>

$$
\begin{equation}
F_{\rm drag}=-c_W\rho_m Av^2,
\label{_auto4} \tag{4}
\end{equation}
$$

and the acceleration is

$$
\begin{eqnarray}
\frac{dv}{dt}=-\frac{c_W\rho_mA}{m}v^2.
\end{eqnarray}
$$

For a particle with initial velocity $v_0$, one can separate the $dt$
to one side of the equation, and move everything with $v$s to the
other side. We did this in our discussion of simple motion and will not repeat it here.

On more general terms,
for many systems, e.g. an automobile, there are multiple sources of
resistance. In addition to wind resistance, where the force is
proportional to $v^2$, there are dissipative effects of the tires on
the pavement, and in the axel and drive train. These other forces can
have components that scale proportional to $v$, and components that
are independent of $v$. Those independent of $v$, e.g. the usual
$f=\mu_K N$ frictional force you consider in your first Physics courses, only set in
once the object is actually moving. As speeds become higher, the $v^2$
components begin to dominate relative to the others. For automobiles
at freeway speeds, the $v^2$ terms are largely responsible for the
loss of efficiency. To travel a distance $L$ at fixed speed $v$, the
energy/work required to overcome the dissipative forces are $fL$,
which for a force of the form $f=\alpha v^n$ becomes

$$
\begin{eqnarray}
W=\int dx~f=\alpha v^n L.
\end{eqnarray}
$$

For $n=0$ the work is
independent of speed, but for the wind resistance, where $n=2$,
slowing down is essential if one wishes to reduce fuel consumption. It
is also important to consider that engines are designed to be most
efficient at a chosen range of power output. Thus, some cars will get
better mileage at higher speeds (They perform better at 50 mph than at
5 mph) despite the considerations mentioned above.


## Going Ballistic, Projectile Motion or a Softer Approach, Falling Raindrops


As an example of Newton's Laws we consider projectile motion (or a
falling raindrop or a ball we throw up in the air) with a drag force. Even though air resistance is
largely proportional to the square of the velocity, we will consider
the drag force to be linear to the velocity, $\boldsymbol{F}=-m\gamma\boldsymbol{v}$,
for the purposes of this exercise. The acceleration for a projectile moving upwards,
$\boldsymbol{a}=\boldsymbol{F}/m$, becomes

$$
\begin{eqnarray}
\frac{dv_x}{dt}=-\gamma v_x,\\
\nonumber
\frac{dv_y}{dt}=-\gamma v_y-g,
\end{eqnarray}
$$

and $\gamma$ has dimensions of inverse time. 

If you on the other hand have a falling raindrop, how do these equations change? See for example Figure 2.1 in Taylor.
Let us stay with a ball which is thrown up in the air at $t=0$. 


## Ways of solving these equations

We will go over two different ways to solve this equation. The first
by direct integration, and the second as a differential equation. To
do this by direct integration, one simply multiplies both sides of the
equations above by $dt$, then divide by the appropriate factors so
that the $v$s are all on one side of the equation and the $dt$ is on
the other. For the $x$ motion one finds an easily integrable equation,

$$
\begin{eqnarray}
\frac{dv_x}{v_x}&=&-\gamma dt,\\
\nonumber
\int_{v_{0x}}^{v_{x}}\frac{dv_x}{v_x}&=&-\gamma\int_0^{t}dt,\\
\nonumber
\ln\left(\frac{v_{x}}{v_{0x}}\right)&=&-\gamma t,\\
\nonumber
v_{x}(t)&=&v_{0x}e^{-\gamma t}.
\end{eqnarray}
$$

This is very much the result you would have written down
by inspection. For the $y$-component of the velocity,

$$
\begin{eqnarray}
\frac{dv_y}{v_y+g/\gamma}&=&-\gamma dt\\
\nonumber
\ln\left(\frac{v_{y}+g/\gamma}{v_{0y}-g/\gamma}\right)&=&-\gamma t_f,\\
\nonumber
v_{fy}&=&-\frac{g}{\gamma}+\left(v_{0y}+\frac{g}{\gamma}\right)e^{-\gamma t}.
\end{eqnarray}
$$

Whereas $v_x$ starts at some value and decays
exponentially to zero, $v_y$ decays exponentially to the terminal
velocity, $v_t=-g/\gamma$.


## Solving as differential equations

Although this direct integration is simpler than the method we invoke
below, the method below will come in useful for some slightly more
difficult differential equations in the future. The differential
equation for $v_x$ is straight-forward to solve. Because it is first
order there is one arbitrary constant, $A$, and by inspection the
solution is

<!-- Equation labels as ordinary links -->
<div id="_auto5"></div>

$$
\begin{equation}
v_x=Ae^{-\gamma t}.
\label{_auto5} \tag{5}
\end{equation}
$$

The arbitrary constants for equations of motion are usually determined
by the initial conditions, or more generally boundary conditions. By
inspection $A=v_{0x}$, the initial $x$ component of the velocity.



## Differential Equations, contn

The differential equation for $v_y$ is a bit more complicated due to
the presence of $g$. Differential equations where all the terms are
linearly proportional to a function, in this case $v_y$, or to
derivatives of the function, e.g., $v_y$, $dv_y/dt$,
$d^2v_y/dt^2\cdots$, are called linear differential equations. If
there are terms proportional to $v^2$, as would happen if the drag
force were proportional to the square of the velocity, the
differential equation is not longer linear. Because this expression
has only one derivative in $v$ it is a first-order linear differential
equation. If a term were added proportional to $d^2v/dt^2$ it would be
a second-order differential equation.  In this case we have a term
completely independent of $v$, the gravitational acceleration $g$, and
the usual strategy is to first rewrite the equation with all the
linear terms on one side of the equal sign,

<!-- Equation labels as ordinary links -->
<div id="_auto6"></div>

$$
\begin{equation}
\frac{dv_y}{dt}+\gamma v_y=-g.
\label{_auto6} \tag{6}
\end{equation}
$$

## Splitting into two parts

Now, the solution to the equation can be broken into two
parts. Because this is a first-order differential equation we know
that there will be one arbitrary constant. Physically, the arbitrary
constant will be determined by setting the initial velocity, though it
could be determined by setting the velocity at any given time. Like
most differential equations, solutions are not "solved". Instead,
one guesses at a form, then shows the guess is correct. For these
types of equations, one first tries to find a single solution,
i.e. one with no arbitrary constants. This is called the {\it
particular} solution, $y_p(t)$, though it should really be called
"a" particular solution because there are an infinite number of such
solutions. One then finds a solution to the {\it homogenous} equation,
which is the equation with zero on the right-hand side,

<!-- Equation labels as ordinary links -->
<div id="_auto7"></div>

$$
\begin{equation}
\frac{dv_{y,h}}{dt}+\gamma v_{y,h}=0.
\label{_auto7} \tag{7}
\end{equation}
$$

Homogenous solutions will have arbitrary constants. 

The particular solution will solve the same equation as the original
general equation

<!-- Equation labels as ordinary links -->
<div id="_auto8"></div>

$$
\begin{equation}
\frac{dv_{y,p}}{dt}+\gamma v_{y,p}=-g.
\label{_auto8} \tag{8}
\end{equation}
$$

However, we don't need find one with arbitrary constants. Hence, it is
called a **particular** solution.

The sum of the two,

<!-- Equation labels as ordinary links -->
<div id="_auto9"></div>

$$
\begin{equation}
v_y=v_{y,p}+v_{y,h},
\label{_auto9} \tag{9}
\end{equation}
$$

is a solution of the total equation because of the linear nature of
the differential equation. One has now found a *general* solution
encompassing all solutions, because it both satisfies the general
equation (like the particular solution), and has an arbitrary constant
that can be adjusted to fit any initial condition (like the homogneous
solution). If the equation were not linear, e.g if there were a term
such as $v_y^2$ or $v_y\dot{v}_y$, this technique would not work.


## More details

Returning to the example above, the homogenous solution is the same as
that for $v_x$, because there was no gravitational acceleration in
that case,

<!-- Equation labels as ordinary links -->
<div id="_auto10"></div>

$$
\begin{equation}
v_{y,h}=Be^{-\gamma t}.
\label{_auto10} \tag{10}
\end{equation}
$$

In this case a particular solution is one with constant velocity,

<!-- Equation labels as ordinary links -->
<div id="_auto11"></div>

$$
\begin{equation}
v_{y,p}=-g/\gamma.
\label{_auto11} \tag{11}
\end{equation}
$$

Note that this is the terminal velocity of a particle falling from a
great height. The general solution is thus,

<!-- Equation labels as ordinary links -->
<div id="_auto12"></div>

$$
\begin{equation}
v_y=Be^{-\gamma t}-g/\gamma,
\label{_auto12} \tag{12}
\end{equation}
$$

and one can find $B$ from the initial velocity,

<!-- Equation labels as ordinary links -->
<div id="_auto13"></div>

$$
\begin{equation}
v_{0y}=B-g/\gamma,~~~B=v_{0y}+g/\gamma.
\label{_auto13} \tag{13}
\end{equation}
$$

Plugging in the expression for $B$ gives the $y$ motion given the initial velocity,

<!-- Equation labels as ordinary links -->
<div id="_auto14"></div>

$$
\begin{equation}
v_y=(v_{0y}+g/\gamma)e^{-\gamma t}-g/\gamma.
\label{_auto14} \tag{14}
\end{equation}
$$

It is easy to see that this solution has $v_y=v_{0y}$ when $t=0$ and
$v_y=-g/\gamma$ when $t\rightarrow\infty$.

One can also integrate the two equations to find the coordinates $x$
and $y$ as functions of $t$,

$$
\begin{eqnarray}
x&=&\int_0^t dt'~v_{0x}(t')=\frac{v_{0x}}{\gamma}\left(1-e^{-\gamma t}\right),\\
\nonumber
y&=&\int_0^t dt'~v_{0y}(t')=-\frac{gt}{\gamma}+\frac{v_{0y}+g/\gamma}{\gamma}\left(1-e^{-\gamma t}\right).
\end{eqnarray}
$$

If the question was to find the position at a time $t$, we would be
finished. However, the more common goal in a projectile equation
problem is to find the range, i.e. the distance $x$ at which $y$
returns to zero. For the case without a drag force this was much
simpler. The solution for the $y$ coordinate would have been
$y=v_{0y}t-gt^2/2$. One would solve for $t$ to make $y=0$, which would
be $t=2v_{0y}/g$, then plug that value for $t$ into $x=v_{0x}t$ to
find $x=2v_{0x}v_{0y}/g=v_0\sin(2\theta_0)/g$. One follows the same
steps here, except that the expression for $y(t)$ is more
complicated. Searching for the time where $y=0$, and we get

<!-- Equation labels as ordinary links -->
<div id="_auto15"></div>

$$
\begin{equation}
0=-\frac{gt}{\gamma}+\frac{v_{0y}+g/\gamma}{\gamma}\left(1-e^{-\gamma t}\right).
\label{_auto15} \tag{15}
\end{equation}
$$

This cannot be inverted into a simple expression $t=\cdots$. Such
expressions are known as "transcendental equations", and are not the
rare instance, but are the norm. In the days before computers, one
might plot the right-hand side of the above graphically as
a function of time, then find the point where it crosses zero.

Now, the most common way to solve for an equation of the above type
would be to apply Newton's method numerically. This involves the
following algorithm for finding solutions of some equation $F(t)=0$.

1. First guess a value for the time, $t_{\rm guess}$.

2. Calculate $F$ and its derivative, $F(t_{\rm guess})$ and $F'(t_{\rm guess})$. 

3. Unless you guessed perfectly, $F\ne 0$, and assuming that $\Delta F\approx F'\Delta t$, one would choose 

4. $\Delta t=-F(t_{\rm guess})/F'(t_{\rm guess})$.

5. Now repeat step 1, but with $t_{\rm guess}\rightarrow t_{\rm guess}+\Delta t$.

If the $F(t)$ were perfectly linear in $t$, one would find $t$ in one
step. Instead, one typically finds a value of $t$ that is closer to
the final answer than $t_{\rm guess}$. One breaks the loop once one
finds $F$ within some acceptable tolerance of zero. A program to do
this will be added shortly.


## Motion in a Magnetic Field


Another example of a velocity-dependent force is magnetism,

$$
\begin{eqnarray}
\boldsymbol{F}&=&q\boldsymbol{v}\times\boldsymbol{B},\\
\nonumber
F_i&=&q\sum_{jk}\epsilon_{ijk}v_jB_k.
\end{eqnarray}
$$

For a uniform field in the $z$ direction $\boldsymbol{B}=B\hat{z}$, the force can only have $x$ and $y$ components,

$$
\begin{eqnarray}
F_x&=&qBv_y\\
\nonumber
F_y&=&-qBv_x.
\end{eqnarray}
$$

The differential equations are

$$
\begin{eqnarray}
\dot{v}_x&=&\omega_c v_y,\omega_c= qB/m\\
\nonumber
\dot{v}_y&=&-\omega_c v_x.
\end{eqnarray}
$$

One can solve the equations by taking time derivatives of either equation, then substituting into the other equation,

$$
\begin{eqnarray}
\ddot{v}_x=\omega_c\dot{v_y}=-\omega_c^2v_x,\\
\nonumber
\ddot{v}_y&=&-\omega_c\dot{v}_x=-\omega_cv_y.
\end{eqnarray}
$$

The solution to these equations can be seen by inspection,

$$
\begin{eqnarray}
v_x&=&A\sin(\omega_ct+\phi),\\
\nonumber
v_y&=&A\cos(\omega_ct+\phi).
\end{eqnarray}
$$

One can integrate the equations to find the positions as a function of time,

$$
\begin{eqnarray}
x-x_0&=&\int_{x_0}^x dx=\int_0^t dt v(t)\\
\nonumber
&=&\frac{-A}{\omega_c}\cos(\omega_ct+\phi),\\
\nonumber
y-y_0&=&\frac{A}{\omega_c}\sin(\omega_ct+\phi).
\end{eqnarray}
$$

The trajectory is a circle centered at $x_0,y_0$ with amplitude $A$ rotating in the clockwise direction.

The equations of motion for the $z$ motion are

<!-- Equation labels as ordinary links -->
<div id="_auto16"></div>

$$
\begin{equation}
\dot{v_z}=0,
\label{_auto16} \tag{16}
\end{equation}
$$

which leads to

<!-- Equation labels as ordinary links -->
<div id="_auto17"></div>

$$
\begin{equation}
z-z_0=V_zt.
\label{_auto17} \tag{17}
\end{equation}
$$

Added onto the circle, the motion is helical.

Note that the kinetic energy,

<!-- Equation labels as ordinary links -->
<div id="_auto18"></div>

$$
\begin{equation}
T=\frac{1}{2}m(v_x^2+v_y^2+v_z^2)=\frac{1}{2}m(\omega_c^2A^2+V_z^2),
\label{_auto18} \tag{18}
\end{equation}
$$

is constant. This is because the force is perpendicular to the
velocity, so that in any differential time element $dt$ the work done
on the particle $\boldsymbol{F}\cdot{dr}=dt\boldsymbol{F}\cdot{v}=0$.

One should think about the implications of a velocity dependent
force. Suppose one had a constant magnetic field in deep space. If a
particle came through with velocity $v_0$, it would undergo cyclotron
motion with radius $R=v_0/\omega_c$. However, if it were still its
motion would remain fixed. Now, suppose an observer looked at the
particle in one reference frame where the particle was moving, then
changed their velocity so that the particle's velocity appeared to be
zero. The motion would change from circular to fixed. Is this
possible?

The solution to the puzzle above relies on understanding
relativity. Imagine that the first observer believes $\boldsymbol{B}\ne 0$ and
that the electric field $\boldsymbol{E}=0$. If the observer then changes
reference frames by accelerating to a velocity $\boldsymbol{v}$, in the new
frame $\boldsymbol{B}$ and $\boldsymbol{E}$ both change. If the observer moved to the
frame where the charge, originally moving with a small velocity $v$,
is now at rest, the new electric field is indeed $\boldsymbol{v}\times\boldsymbol{B}$,
which then leads to the same acceleration as one had before. If the
velocity is not small compared to the speed of light, additional
$\gamma$ factors come into play,
$\gamma=1/\sqrt{1-(v/c)^2}$. Relativistic motion will not be
considered in this course.




## Sliding Block tied to a Wall

Another classical case is that of simple harmonic oscillations, here represented by a block sliding on a horizontal frictionless surface. The block is tied to a wall with a spring. If the spring is not compressed or stretched too far, the force on the block at a given position $x$ is

$$
F=-kx.
$$

The negative sign means that the force acts to restore the object to an equilibrium position. Newton's equation of motion for this idealized system is then

$$
m\frac{d^2x}{dt^2}=-kx,
$$

or we could rephrase it as

<!-- Equation labels as ordinary links -->
<div id="eq:newton1"></div>

$$
\frac{d^2x}{dt^2}=-\frac{k}{m}x=-\omega_0^2x,
\label{eq:newton1} \tag{19}
$$

with the angular frequency $\omega_0^2=k/m$. 

The above differential equation has the advantage that it can be solved  analytically with solutions on the form

$$
x(t)=Acos(\omega_0t+\nu),
$$

where $A$ is the amplitude and $\nu$ the phase constant.   This provides in turn an important test for the numerical
solution and the development of a program for more complicated cases which cannot be solved analytically. 


With the position $x(t)$ and the velocity  $v(t)=dx/dt$ we can reformulate Newton's equation in the following way

$$
\frac{dx(t)}{dt}=v(t),
$$

and

$$
\frac{dv(t)}{dt}=-\omega_0^2x(t).
$$

We are now going to solve these equations using first the standard forward Euler  method. Later we will try to improve upon this.


Before proceeding however, it is important to note that in addition to the exact solution, we have at least two further tests which can be used to check our solution. 

Since functions like $cos$ are periodic with a period $2\pi$, then the solution $x(t)$ has also to be periodic. This means that

$$
x(t+T)=x(t),
$$

with $T$ the period defined as

$$
T=\frac{2\pi}{\omega_0}=\frac{2\pi}{\sqrt{k/m}}.
$$

Observe that $T$ depends only on $k/m$ and not on the amplitude of the solution. 


In addition to the periodicity test, the total energy has also to be conserved. 

Suppose we choose the initial conditions

$$
x(t=0)=1\hspace{0.1cm} \mathrm{m}\hspace{1cm} v(t=0)=0\hspace{0.1cm}\mathrm{m/s},
$$

meaning that block is at rest at $t=0$ but with a potential energy

$$
E_0=\frac{1}{2}kx(t=0)^2=\frac{1}{2}k.
$$

The total energy at any time $t$ has however to be conserved, meaning that our solution has to fulfil the condition

$$
E_0=\frac{1}{2}kx(t)^2+\frac{1}{2}mv(t)^2.
$$

We will derive this equation in our discussion on [energy conservation](https://mhjensen.github.io/Physics321/doc/pub/energyconserv/html/energyconserv.html).


An algorithm which implements these equations is included below.
 * Choose the initial position and speed, with the most common choice $v(t=0)=0$ and some fixed value for the position. 

 * Choose the method you wish to employ in solving the problem.

 * Subdivide the time interval $[t_i,t_f] $ into a grid with step size

$$
h=\frac{t_f-t_i}{N},
$$

where $N$ is the number of mesh points. 
 * Calculate now the total energy given by

$$
E_0=\frac{1}{2}kx(t=0)^2=\frac{1}{2}k.
$$

* Choose ODE solver to obtain $x_{i+1}$ and $v_{i+1}$ starting from the previous values $x_i$ and $v_i$.

 * When we have computed $x(v)_{i+1}$ we upgrade  $t_{i+1}=t_i+h$.

 * This iterative  process continues till we reach the maximum time $t_f$.

 * The results are checked against the exact solution. Furthermore, one has to check the stability of the numerical solution against the chosen number of mesh points $N$.      

The following python program ( code will be added shortly)

#
# This program solves Newtons equation for a block sliding on
# an horizontal frictionless surface.
# The block is tied to the wall with a spring, so N's eq takes the form:
#
#  m d^2x/dt^2 = - kx
#
# In order to make the solution dimless, we set k/m = 1.
# This results in two coupled diff. eq's that may be written as:
#
#  dx/dt = v
#  dv/dt = -x
#
# The user has to specify the initial velocity and position,
# and the number of steps. The time interval is fixed to
# t \in [0, 4\pi) (two periods)
#

## The classical pendulum and scaling the equations

The angular equation of motion of the pendulum is given by
Newton's equation and with no external force it reads

<!-- Equation labels as ordinary links -->
<div id="_auto19"></div>

$$
\begin{equation}
  ml\frac{d^2\theta}{dt^2}+mgsin(\theta)=0,
\label{_auto19} \tag{20}
\end{equation}
$$

with an angular velocity and acceleration given by

<!-- Equation labels as ordinary links -->
<div id="_auto20"></div>

$$
\begin{equation}
     v=l\frac{d\theta}{dt},
\label{_auto20} \tag{21}
\end{equation}
$$

and

<!-- Equation labels as ordinary links -->
<div id="_auto21"></div>

$$
\begin{equation}
     a=l\frac{d^2\theta}{dt^2}.
\label{_auto21} \tag{22}
\end{equation}
$$

## More on the Pendulum

We do however expect that the motion will gradually come to an end due a viscous drag torque acting on the pendulum. 
In the presence of the drag, the above equation becomes

<!-- Equation labels as ordinary links -->
<div id="eq:pend1"></div>

$$
\begin{equation}
   ml\frac{d^2\theta}{dt^2}+\nu\frac{d\theta}{dt}  +mgsin(\theta)=0, \label{eq:pend1} \tag{23}
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
   ml\frac{d^2\theta}{dt^2}+\nu\frac{d\theta}{dt}  +mgsin(\theta)=Asin(\omega t), \label{eq:pend2} \tag{24}
\end{equation}
$$

with $A$ and $\omega$ two constants representing the amplitude and 
the angular frequency respectively. The latter is called the driving frequency.




## More on the Pendulum

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

These are the equations to be solved.  The factor $Q$ represents the number of oscillations of the undriven system that must occur before  its energy is significantly reduced due to the viscous drag. The  amplitude $\hat{A}$ is measured in units of the maximum possible  gravitational torque while $\hat{\omega}$ is the angular frequency of the external torque measured in units of the pendulum's natural frequency.