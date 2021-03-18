For linux users, compile and link with 
c++ -O3 -fopenmp -o <executable>  <nameofprogram>

For OS X users, you need to install clang-omp using brew, that is
brew install libomp

To compile under OS X, if you are using brew to install libomp, you
may, unless you have defined the path, need to compile as

c++ -O3 -o <executable>  <programfile> -I/opt/homebrew/include/  -L/opt/homebrew/lib/  -lomp

If you have set these paths (homebrew installs now by default uner /opt/homebrew)  you can compile and link with

c++ -O3 -o <executable>  <programfile> -lomp



 
