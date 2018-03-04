#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <armadillo>

using namespace  std;
using namespace arma;

vec SteepestDescent(mat A, vec b, vec x0, int dim){
  int IterMax, i;
  const double tolerance = 1.0e-14;
  vec x(dim),r(dim),t(dim);
  double c ,alpha,d;
  IterMax = 20;
  x = x0;
  r = A*x-b;
  i = 0;
  while (i <= IterMax || sqrt(dot(r,r)) < tolerance ){
    if(sqrt(dot(r,r))<tolerance){
       cerr << "An error has occurred: execution of function terminated" << endl;
       break;
    }
    t = A*r;
    c = dot(r,r);
    alpha = c/dot(r,t);
    x = x - alpha*r;
    r =  A*x-b;
    if(sqrt(dot(r,r)) < tolerance) break;
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

  xsd = SteepestDescent(A,b,x0, dim);
  

  cout << "The approximate solution using Steepest Descent is: " << endl;
  xsd.print();
  cout << endl;


 
}




