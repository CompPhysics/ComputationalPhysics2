# Textbooks and practicalities


## Textbooks

_Recommended textbooks_: Lecture Notes by Morten Hjorth-Jensen and Bernd A. Berg, Markov Chain Monte Carlo Simulations and their Statistical Analysis, World Scientific, 2004, chapters 1, 2


## Practicalities

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



### How do I run MPI/OpneMP on a PC/Laptop? MPI
To install MPI is rather easy on hardware running unix/linux as operating systems, follow simply the instructions from the "OpenMPI website":"https://www.open-mpi.org/". See also subsequent slides.
When you have made sure you have installed MPI on your PC/laptop, 
- Compile with mpicxx/mpic++ or mpif90
  Compile and link
  mpic++ -O3 -o nameofprog.x nameofprog.cpp
  run code with for example 8 processes using mpirun/mpiexec
  mpiexec -n 8 ./nameofprog.x

If you wish to install MPI and OpenMP 
on your laptop/PC, we recommend the following:


- For OpenMP, the compile option _-fopenmp_ is included automatically in recent versions of the C++ compiler and Fortran compilers. For users of different Linux distributions, simply use the available C++ or Fortran compilers and add the above compiler instructions, see also code examples below.
- For OS X users however, install _libomp_
  brew install libomp
and compile and link as
c++ -o <name executable> <name program.cpp>  -lomp

For linux/ubuntu users, you need to install two packages (alternatively use the synaptic package manager)
  sudo apt-get install libopenmpi-dev
  sudo apt-get install openmpi-bin

For OS X users, install brew (after having installed xcode and gcc, needed for the 
gfortran compiler of openmpi) and then install with brew

   brew install openmpi

When running an executable (code.x), run as

  mpirun -n 10 ./code.x

where we indicate that we want  the number of processes to be 10.




