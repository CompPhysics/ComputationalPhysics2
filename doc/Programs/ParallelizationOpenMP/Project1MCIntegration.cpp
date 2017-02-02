// OpenMP code for numerical integration with Monte Carlo importance sampling
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <random>
#include  <omp.h>
using namespace std; 

// The integrand in Cartesian coordinates
double Integrand(double *);
// The Monte Carlo integration with Gaussian integration points
double  MonteCarloGaussianIntegration(int);

// Start main program
int main (int argc, char* argv[])
{
  // read from terminal the number of integration points
  int n = atoi(argv[1]);
  int thread_num;

  double wtime;
  cout << "  Compute 4d-integral using Gaussian quadrature with Hermite quadrature." << endl;
  omp_set_num_threads(4);
  thread_num = omp_get_max_threads ();
  cout << "  The number of processors available = " << omp_get_num_procs () << endl ;
  cout << "  The number of threads available    = " << thread_num <<  endl;
  cout << "  The number of integration points                 = " << n << endl;
  wtime = omp_get_wtime ( );
  double Integral = MonteCarloGaussianIntegration(n);
  wtime = omp_get_wtime ( ) - wtime;
  cout << setiosflags(ios::showpoint | ios::uppercase);
  cout << setprecision(15) << setw(20) << "Time used  integration=" << wtime  << endl;
  cout << "Final Integral    = " << Integral << endl;
  return 0;
} // end main function

// The integrand, here a first order Hermite polynomial for x1 only
// The other variables are given by zeroth-order Hermite polynomials.
double Integrand(double *x)
{
  return (2*x[0])*(2*x[0]);
}

//Plain Gauss-Hermite integration with cartesian variables, brute force
double  MonteCarloGaussianIntegration(int n)
{
  double x[4];
  // Setting the normal distribution
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);
  double Integral = 0.0;
  double jacobidet = pow((acos(-1.)),2.);
  int i, j;
  long idum=-1; 
# pragma omp parallel default(shared) private (i, j) reduction(+:Integral)
  {
# pragma omp for
     for (i = 1;  i <= n; i++){
//   x[] contains the random numbers for all dimensions
       for (j = 0; j < 4; j++) {
	 x[j]=distribution(generator);
       }
       Integral += Integrand(x);
     }
     Integral = jacobidet*Integral/((double) n );
  } // end parallel region
  return Integral;
}

