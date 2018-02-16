#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>

using namespace  std;
using namespace arma;

vec ConjugateGradient(mat A, vec b, vec x0, int dim){
  int IterMax, i;
  const double tolerance = 1.0e-14;
  vec x(dim),r(dim),v(dim),z(dim);
  double c,t,d;

  IterMax = dim;
  x = x0;
  r = b - A*x;
  v = r;
  c = dot(r,r);
  i = 0;
  while (i <= IterMax || sqrt(dot(v,v)) < tolerance ){
    if(sqrt(dot(v,v))<tolerance){
      cerr << "An error has occurred in ConjugateGradient: execution of function terminated" << endl;
      break;
    }
    z = A*v;
    t = c/dot(v,z);
    x = x + t*v;
    r = r - t*z;
    d = dot(r,r);
    if(sqrt(d) < tolerance)
      break;
    v = r + (d/c)*v;
    c = d;
    i++;
  }
  return x;
} 



//   Main function begins here
int main(int  argc, char * argv[]){
  int dim = 2;
  vec x(dim),xsd(dim), b(dim),x0(dim);
  mat A(dim,dim);
  
  // Set our initial guess
  x0(0) = x0(1) = 0;
  // Set the matrix  
  A(0,0) =  3;    A(1,0) =  2;   A(0,1) =  2;   A(1,1) =  6; 
  
  cout << "The Matrix A that we are using: " << endl;
  A.print();
  cout << endl;

  vec y(dim);
  y(0) = 2.;
  y(1) = -2.;

  cout << "The exact solution is: " << endl;
  y.print();
  cout << endl;
  b = A*y;

  cout << "The right hand side, b, of the expression Ax=b: " << endl;
  b.print();
  cout << endl;

  xsd = ConjugateGradient(A,b,x0, dim);
  

  cout << "The approximate solution using Conjugate Gradient is: " << endl;
  xsd.print();
  cout << endl;


 
}



