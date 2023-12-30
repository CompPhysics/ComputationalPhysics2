# Computational Physics 2: computational quantum mechanics

## Introduction

This repository contain lecture slides, programs, exercises and
projects for a more advanced course in computational physics, with an
emphasis on quantum mechanical problems with many interacting
particles. The applications and the computational methods are relevant
for research problems in such diverse areas as nuclear, atomic,
molecular and solid-state physics, quantum chemistry and materials
science.

A theoretical understanding of the behavior of quantum-mechanical
many-body systems - that is, systems containing many interacting
particles - is a considerable challenge in that no exact solution can
be found; instead, reliable methods are needed for approximate but
accurate simulations of such systems on modern computers. New insights
and a better understanding of complicated quantum mechanical systems
can only be obtained via large-scale simulations. The capability to
study such systems is of high relevance for both fundamental research
and industrial and technological advances.

The aim of this course is to present applications of, through various
computational projects, some of the most widely used many-body methods
with pertinent algorithms and high-performance computing topics such
as advanced parallelization techniques and object
orientation. Furthermore, Machine Learning and quantum
computing may be presented if of interest. 
The methods and algorithms that will be studied
may vary from year to year depending on the interests of the
participants, but the main focus will be on systems from computational
material science, solid-state physics, atomic and molecular physics,
nuclear physics and quantum chemistry. The most relevant algorithms
and methods are microscopic mean-field theories (Hartree-Fock and
Kohn-Sham theories and density functional theories), large-scale
diagonalization methods, coupled-cluster theory, similarity
renormalization methods, and quantum Monte Carlo like Variational
Monte Carlo and Diffusion Monte Carlo approaches. Methods to study
phase transitions for both fermionic and bosonic systems can also be
studied.




## Learning outcomes

The course introduces a variety of central algorithms and methods for
professional studies of quantum mechanical systems, with relevance for
several problems in physics, materials science and quantum
chemistry. The course is project based and through the various
projects, normally two, the participants will be exposed to
fundamental research problems in these fields, with the aim to
reproduce state of the art scientific results. The students will learn
to develop and structure large codes for studying these systems, get
aquainted with supercomputing facilities and learn to handle large
scientific projects. A good scientific and ethical conduct is
emphasized throughout the course.

The course is also a continuation of FYS3150 – Computational Physics,
and it will give a further treatment of several of the numerical
methods given there.


## Prerequisites

Basic knowledge in programming and mathematics, with an emphasis on
linear algebra. Knowledge of Python or/and C++ as programming
languages is strongly recommended and experience with Jupiter notebook
is recommended. Required courses are the equivalents to the University
of Oslo mathematics courses MAT1100, MAT1110, MAT1120 and at least one
of the corresponding computing and programming courses INF1000/INF1110
or MAT-INF1100/MAT-INF1100L/BIOS1100/KJM-INF1100. Most universities
offer nowadays a basic programming course (often compulsory) where
Python is the recurring programming language.

We recommend also that you have some background in quantum mechanics, typically at the level of FYS2140 and/or FYS3110.

## The course has two central parts

Computational aspects play a central role and you are expected to work
on numerical examples and projects which illustrate the theory and
varous algorithms discussed during the lectures. We recommend strongly
to form small project groups of 2-3 participants, if possible.

## Instructor information
* _Name_: Morten Hjorth-Jensen
* _Email_: morten.hjorth-jensen@fys.uio.no
* _Phone_: +47-48257387
* _Office_: Department of Physics, University of Oslo, Eastern wing, room FØ470 
* _Office hours_: *Anytime*! 


##  Teaching Assistants Spring Semester 2022
* _Name_:  Øyvind Sigmundson Schøyen, 
* _Email_: oyvinssc@student.matnat.uio.no 	 
* _Office_: Department of Physics, University of Oslo, Eastern wing, room FØ452

## Practicalities

1. Two lectures per week, spring semester, 10 ECTS. The lectures are recorded and linked to this site and the official University of Oslo website for the course;
2. Thre hours of laboratory sessions for work on computational projects and exercises. There will  also be fully digital laboratory sessions for those who cannot attend in person;
3. Two projects which are graded and count 1/2 each of the final grade;
4. The course is offered as a FYS4411 (Master of Science level) and a FYS9411 (PhD level) course;
5. Weekly emails with summary of activities will be mailed to all participants;

## Grading
Grading scale: Grades are awarded on a scale from A to F, where A is the best grade and F is a fail. There are two projects which are graded and each project counts 1/2 of the final grade. The total score is thus the average from the two  projects.

The final number of points is based on the average of all projects (including eventual additional points) and the grade follows the following table:

 * 92-100 points: A
 * 77-91 points: B
 * 58-76 points: C
 * 46-57 points: D
 * 40-45 points: E
 * 0-39 points: F-failed

## Required Technologies

Course participants are expected to have their own laptops/PCs. We use _Git_ as version control software and the usage of providers like _GitHub_, _GitLab_ or similar are strongly recommended.

We will make extensive use of Python and C++ as programming languages.


If you have Python installed and you feel
pretty familiar with installing different packages, we recommend that
you install the following Python packages via _pip_ as 

* pip install numpy scipy matplotlib ipython scikit-learn mglearn sympy pandas pillow 

For OSX users we recommend, after having installed Xcode, to
install _brew_. Brew allows for a seamless installation of additional
software via for example 

* brew install python3

For Linux users, with its variety of distributions like for example the widely popular Ubuntu distribution,
you can use _pip_ as well and simply install Python as 

* sudo apt-get install python3

### Python installers

If you don't want to perform these operations separately and venture
into the hassle of exploring how to set up dependencies and paths, we
recommend two widely used distrubutions which set up all relevant
dependencies for Python, namely 

* Anaconda:https://docs.anaconda.com/, 

which is an open source
distribution of the Python and R programming languages for large-scale
data processing, predictive analytics, and scientific computing, that
aims to simplify package management and deployment. Package versions
are managed by the package management system _conda_. 

* Enthought canopy:https://www.enthought.com/product/canopy/ 

is a Python
distribution for scientific and analytic computing distribution and
analysis environment, available for free and under a commercial
license.

Furthermore, Google's Colab:https://colab.research.google.com/notebooks/welcome.ipynb is a free Jupyter notebook environment that requires 
no setup and runs entirely in the cloud. Try it out!

### Useful Python libraries
Here we list several useful Python libraries we strongly recommend (if you use anaconda many of these are already there)

* _NumPy_:https://www.numpy.org/ is a highly popular library for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
* _The pandas_:https://pandas.pydata.org/ library provides high-performance, easy-to-use data structures and data analysis tools 
* _Xarray_:http://xarray.pydata.org/en/stable/ is a Python package that makes working with labelled multi-dimensional arrays simple, efficient, and fun!
* _Scipy_:https://www.scipy.org/ (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering. 
* _Matplotlib_:https://matplotlib.org/ is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.
* _Autograd_:https://github.com/HIPS/autograd can automatically differentiate native Python and Numpy code. It can handle a large subset of Python's features, including loops, ifs, recursion and closures, and it can even take derivatives of derivatives of derivatives
* _SymPy_:https://www.sympy.org/en/index.html is a Python library for symbolic mathematics. 
* _scikit-learn_:https://scikit-learn.org/stable/ has simple and efficient tools for machine learning, data mining and data analysis
* _TensorFlow_:https://www.tensorflow.org/ is a Python library for fast numerical computing created and released by Google
* _Keras_:https://keras.io/ is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano
* And many more such as _pytorch_:https://pytorch.org/,  _Theano_:https://pypi.org/project/Theano/ etc 

###  Useful C++ libraries

* _Armadillo_: http://arma.sourceforge.net/ Armadillo is a high quality linear algebra library (matrix maths) for the C++ language, aiming towards a good balance between speed and ease of use
* _Eigen_: http://eigen.tuxfamily.org/index.php?title=Main_Page Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
* _Scythe_: http://scythe.lsa.umich.edu/ The Scythe Statistical Library is an open source C++ library for statistical computation
* _Autodiff_: https://autodiff.github.io/ autodiff is a C++17 library that uses modern and advanced programming techniques to enable automatic computation of derivatives in an efficient and easy way.
* _Optmilib_: https://www.kthohr.com/optimlib.html OptimLib is a lightweight C++ library of numerical optimization methods for nonlinear functions.

## Textbooks

_Recommended textbooks_: Lecture Notes by Morten Hjorth-Jensen

