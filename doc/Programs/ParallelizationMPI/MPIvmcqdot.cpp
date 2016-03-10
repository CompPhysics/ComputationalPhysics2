// Variational Monte Carlo for atoms with importance sampling, slater det
// Test case for 2-electron quantum dot
#include "mpi.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "vectormatrixclass.h"

using namespace  std;
// output file as global variable
ofstream ofile;  
// the step length and its squared inverse for the second derivative 
//  Here we define global variables  used in various functions
//  These can be changed by reading from file the different parameters
int dimension = 2; // three-dimensional system
int my_rank, numprocs;  //  these are the parameters used by MPI  to define which node and how many
double timestep = 0.05;  //  we fix the time step  for the gaussian deviate
int number_particles  = 2;  //  we fix also the number of electrons to be 2

// declaration of functions 

// The Mc sampling for the variational Monte Carlo 
void  mc_sampling(int, double &, double &, Vector &);

// The variational wave function
double  wave_function(Matrix &, Vector &);

// The local energy 
double  local_energy(Matrix &, Vector &);

// The quantum force
void  quantum_force(Matrix &, Matrix &, Vector &);

// ran2 for uniform deviates, initialize with negative seed.
double ran2(long *);

// function for gaussian random numbers
double gaussian_deviate(long *);

// inline function for single-particle wave function
inline double psi1s(double r, double alpha) { 
   return exp(-alpha*r*0.5);
}

// inline function for derivative of single-particle wave function
inline double deriv_spwf(double r, double alpha) { 
  return -r*alpha;
}

// function for absolute value of relative distance
double abs_relative(Matrix &r, int i, int j) { 
      double r_ij = 0;  
      for (int k = 0; k < dimension; k++) { 
	r_ij += (r(i,k)-r(j,k))*(r(i,k)-r(j,k));
      }
      return sqrt(r_ij); 
}

// inline function for derivative of Jastrow factor
inline double deriv_jastrow(Matrix &r, double beta, int i, int j, int k){
  return (r(i,k)-r(j,k))/(abs_relative(r, i, j)*pow(1.0+beta*abs_relative(r, i, j),2));
}

// function for square of position of single particle
double singleparticle_pos2(Matrix &r, int i) { 
    double r_single_particle = 0;
    for (int j = 0; j < dimension; j++) { 
      r_single_particle  += r(i,j)*r(i,j);
    }
    return r_single_particle;
}

void lnsrch(int n, Vector &xold, double fold, Vector &g, Vector &p, Vector &x,
		 double *f, double stpmax, int *check, double (*func)(Vector &p));

void dfpmin(Vector &p, int n, double gtol, int *iter, double *fret,
	    double(*func)(Vector &p), void (*dfunc)(Vector &p, Vector &g));

static double sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)


static double maxarg1,maxarg2;
#define FMAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
        (maxarg1) : (maxarg2))


// Begin of main program   

int main(int argc, char* argv[])
{
  char *outfilename;
  int i;
  double total_number_cycles;
  double  time_start, time_end, total_time;
  int number_cycles = 100000; //  default number of cycles
  double variance, energy, error;
  double total_cumulative_e, total_cumulative_e2, cumulative_e, cumulative_e2;
  //  MPI initializations
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  time_start = MPI_Wtime();

  if (my_rank == 0 && argc <= 1) {
    cout << "Bad Usage: " << argv[0] << 
      " read also output file on same line" << endl;
  }
  if (my_rank == 0 && argc > 1) {
    outfilename=argv[1];
    ofile.open(outfilename); 
  }
  Vector variate(2);
  variate(0) = 1.0;  // value of alpha
  variate(1) = 0.4;  // value of beta
  // broadcast the total number of  variations
  //  MPI_Bcast (&number_cycles, 1, MPI_INT, 0, MPI_COMM_WORLD);
  total_number_cycles = number_cycles*numprocs; 
  //  Do the mc sampling  and accumulate data with MPI_Reduce
  cumulative_e = cumulative_e2 = 0.0;
  mc_sampling(number_cycles, cumulative_e, cumulative_e2, variate);
  //  Collect data in total averages
  MPI_Reduce(&cumulative_e, &total_cumulative_e, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&cumulative_e2, &total_cumulative_e2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  time_end = MPI_Wtime();
  total_time = time_end-time_start;
  // Print out results  
  if ( my_rank == 0) {
    cout << "Time = " <<  total_time  << " on number of processors: "  << numprocs  << endl;
    energy = total_cumulative_e/numprocs;
    variance = total_cumulative_e2/numprocs-energy*energy;
    error=sqrt(variance/(total_number_cycles-1.0));
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << variate(0);
    ofile << setw(15) << setprecision(8) << variate(1);
    ofile << setw(15) << setprecision(8) << energy;
    ofile << setw(15) << setprecision(8) << variance;
    ofile << setw(15) << setprecision(8) << error << endl;
    ofile.close();  // close output file
  }
  // End MPI
  MPI_Finalize ();  
  return 0;
}  //  end of main function


// Monte Carlo sampling with the Metropolis algorithm  

void mc_sampling(int number_cycles, double &cumulative_e, double &cumulative_e2, Vector &variate)
{
  int cycles, i, j, k;
  long idum;
  double wfnew, wfold, energy, energy2, delta_e;
  double greensfunction, D; 
  // diffusion constant from Schroedinger equation
  D = 0.5; 
  // every node has its own seed for the random numbers
  idum = -1-my_rank;
  // allocate matrices which contain the position of the particles  
  Matrix r_old( number_particles, dimension), r_new( number_particles, dimension);
  Matrix qforce_old(number_particles, dimension), qforce_new(number_particles, dimension);
  energy = energy2 = delta_e = 0.0;
  //  initial trial positions
  for (i = 0; i < number_particles; i++) { 
    for ( j=0; j < dimension; j++) {
      r_old(i,j) = gaussian_deviate(&idum)*sqrt(timestep);
    }
  }
  wfold = wave_function(r_old, variate);
  quantum_force(r_old, qforce_old, variate);
  // loop over monte carlo cycles 
  for (cycles = 1; cycles <= number_cycles; cycles++){ 
    // new position 
    for (i = 0; i < number_particles; i++) { 
      for ( j=0; j < dimension; j++) {
	// gaussian deviate to compute new positions using a given timestep
	r_new(i,j) = r_old(i,j) + gaussian_deviate(&idum)*sqrt(timestep)+qforce_old(i,j)*timestep*D;
      }  
      //  for the other particles we need to set the position to the old position since
      //  we move only one particle at the time
      for (k = 0; k < number_particles; k++) {
	if ( k != i) {
	  for ( j=0; j < dimension; j++) {
	    r_new(k,j) = r_old(k,j);
	  }
	} 
      }
      wfnew = wave_function(r_new, variate); 
      quantum_force(r_new, qforce_new, variate);
      //  we compute the log of the ratio of the greens functions to be used in the 
      //  Metropolis-Hastings algorithm
      greensfunction = 0.0;            
      for ( j=0; j < dimension; j++) {
	greensfunction += 0.5*(qforce_old(i,j)+qforce_new(i,j))*
	  (D*timestep*0.5*(qforce_old(i,j)-qforce_new(i,j))-r_new(i,j)+r_old(i,j));
      }
      greensfunction = exp(greensfunction);
      // The Metropolis test is performed by moving one particle at the time
      if(ran2(&idum) <= greensfunction*wfnew*wfnew/wfold/wfold ) { 
	for ( j=0; j < dimension; j++) {
	  r_old(i,j) = r_new(i,j);
	  qforce_old(i,j) = qforce_new(i,j);
	}
	wfold = wfnew;
      }
    }  //  end of loop over particles
    // compute local energy  
    delta_e = local_energy(r_old, variate);
    // update energies
    energy += delta_e;
    energy2 += delta_e*delta_e;
  }   // end of loop over MC trials   
  // update the energy average and its squared 
  cumulative_e = energy/number_cycles;
  cumulative_e2 = energy2/number_cycles;
}   // end mc_sampling function  


// Function to compute the squared wave function and the quantum force

double  wave_function(Matrix &r, Vector &variate)
{
  int i, j;
  double wf;
  wf = 0.0;
  // full Slater determinant for two particles, 
  wf  = psi1s(singleparticle_pos2(r, 0), variate(0))*psi1s(singleparticle_pos2(r, 1),variate(0));
  // contribution from Jastrow factor
  for (i = 0; i < number_particles-1; i++) { 
    for (j = i+1; j < number_particles; j++) {
      wf *= exp(abs_relative(r, i, j)/((1.0+variate(1)*abs_relative(r, i, j))));
    }
  }
  return wf;
}

// Function to calculate the local energy with num derivative

double  local_energy(Matrix &r, Vector &variate)
{
  int i, j , k;
  double e_local, e_kinetic, e_potential, sum1;
  // compute the kinetic and potential energy from the single-particle part
  // for a many-electron system this has to be replaced by a Slater determinant

  // The absolute value of the interparticle length
  Matrix length( number_particles, number_particles);
  // Set up interparticle distance
  for (i = 0; i < number_particles-1; i++) { 
    for(j = i+1; j < number_particles; j++){
      length(i,j) = abs_relative(r, i, j);
      length(j,i) =  length(i,j);
    }
  }

  e_kinetic = 0.0;
  // Set up kinetic energy from Slater and Jastrow terms
  for (i = 0; i < number_particles; i++) { 
    for (k = 0; k < dimension; k++) {
      sum1 = 0.0; 
      for(j = 0; j < number_particles; j++){
	if ( j != i) {
	  sum1 += deriv_jastrow(r, variate(1), i, j, k);
	}
      }
      e_kinetic += (sum1+deriv_spwf(r(i,k),variate(0)))*(sum1+deriv_spwf(r(i,k),variate(0)));
    }
  }
  e_kinetic += -2*variate(0)*number_particles;
  for (int i = 0; i < number_particles-1; i++) {
      for (int j = i+1; j < number_particles; j++) {
        e_kinetic += 2.0/(pow(1.0 + variate(1)*length(i,j),2))*(1.0/length(i,j)-2*variate(1)/(1+variate(1)*length(i,j)) );
      }
  }
  e_kinetic *= -0.5;

  e_potential = 0;
  for (i = 0; i < number_particles; i++) { 
    double rsq = singleparticle_pos2(r, i);
    e_potential += 0.5*rsq;  // sp energy HO part
  }
  // finally just the electron-electron repulsion
  for (i = 0; i < number_particles-1; i++) { 
    for (j = i+1; j < number_particles; j++) {
      e_potential += 1.0/length(i,j);          
    }
  }
  e_local = e_kinetic+e_potential;
  return e_local;
}

void  quantum_force(Matrix &r, Matrix &qforce, Vector &variate)
{
  int i, j, k;
  double sppart, Jsum;
  // compute the first derivative 
  for (i = 0; i < number_particles; i++) {
    for (k = 0; k < dimension; k++) {
      // single-particle part
      sppart = deriv_spwf(r(i,k),variate(0));
      //  Jastrow factor contribution
      Jsum = 0.0;
      for (j = 0; j < number_particles; j++) {
	if ( j != i) {
	  Jsum += deriv_jastrow(r, variate(1), i, j, k);
	}
      }
      qforce(i,k) = 2*(Jsum+sppart);
    }
  }
} // end of quantum_force function


/*
** The function 
**         ran2()
** is a long periode (> 2 x 10^18) random number generator of 
** L'Ecuyer and Bays-Durham shuffle and added safeguards.
** Call with idum a negative integer to initialize; thereafter,
** do not alter idum between sucessive deviates in a
** sequence. RNMX should approximate the largest floating point value
** that is less than 1.
** The function returns a uniform deviate between 0.0 and 1.0
** (exclusive of end-point values).
*/

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran2(long *idum)
{
  int            j;
  long           k;
  static long    idum2 = 123456789;
  static long    iy=0;
  static long    iv[NTAB];
  double         temp;

  if(*idum <= 0) {
    if(-(*idum) < 1) *idum = 1;
    else             *idum = -(*idum);
    idum2 = (*idum);
    for(j = NTAB + 7; j >= 0; j--) {
      k     = (*idum)/IQ1;
      *idum = IA1*(*idum - k*IQ1) - k*IR1;
      if(*idum < 0) *idum +=  IM1;
      if(j < NTAB)  iv[j]  = *idum;
    }
    iy=iv[0];
  }
  k     = (*idum)/IQ1;
  *idum = IA1*(*idum - k*IQ1) - k*IR1;
  if(*idum < 0) *idum += IM1;
  k     = idum2/IQ2;
  idum2 = IA2*(idum2 - k*IQ2) - k*IR2;
  if(idum2 < 0) idum2 += IM2;
  j     = iy/NDIV;
  iy    = iv[j] - idum2;
  iv[j] = *idum;
  if(iy < 1) iy += IMM1;
  if((temp = AM*iy) > RNMX) return RNMX;
  else return temp;
}
#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

// End: function ran2()



// random numbers with gaussian distribution
double gaussian_deviate(long * idum)
{
  static int iset = 0;
  static double gset;
  double fac, rsq, v1, v2;

  if ( idum < 0) iset =0;
  if (iset == 0) {
    do {
      v1 = 2.*ran2(idum) -1.0;
      v2 = 2.*ran2(idum) -1.0;
      rsq = v1*v1+v2*v2;
    } while (rsq >= 1.0 || rsq == 0.);
    fac = sqrt(-2.*log(rsq)/rsq);
    gset = v1*fac;
    iset = 1;
    return v2*fac;
  } else {
    iset =0;
    return gset;
  }
} // end function for gaussian deviates


#define ITMAX 200
#define EPS 3.0e-8
#define TOLX (4*EPS)
#define STPMX 100.0


void dfpmin(Vector &p, int n, double gtol, int *iter, double *fret,
	    double(*func)(Vector &p), void (*dfunc)(Vector &p, Vector &g))
{

  int check,i,its,j;
  double den,fac,fad,fae,fp,stpmax,sum=0.0,sumdg,sumxi,temp,test;
  Vector dg(n), g(n), hdg(n), pnew(n), xi(n);
  Matrix hessian(n,n);

  fp=(*func)(p);
  (*dfunc)(p,g);
  for (i = 0;i < n;i++) {
    for (j = 0; j< n;j++) hessian(i,j)=0.0;
    hessian(i,i)=1.0;
    xi(i) = -g(i);
    sum += p(i)*p(i);
  }
  stpmax=STPMX*FMAX(sqrt(sum),(double)n);
  for (its=1;its<=ITMAX;its++) {
    *iter=its;
    lnsrch(n,p,fp,g,xi,pnew,fret,stpmax,&check,func);
    fp = *fret;
    for (i = 0; i< n;i++) {
      xi(i)=pnew(i)-p(i);
      p(i)=pnew(i);
    }
    test=0.0;
    for (i = 0;i< n;i++) {
      temp=fabs(xi(i))/FMAX(fabs(p(i)),1.0);
      if (temp > test) test=temp;
    }
    if (test < TOLX) {
      return;
    }
    for (i=0;i<n;i++) dg(i)=g(i);
    (*dfunc)(p,g);
    test=0.0;
    den=FMAX(*fret,1.0);
    for (i=0;i<n;i++) {
      temp=fabs(g(i))*FMAX(fabs(p(i)),1.0)/den;
      if (temp > test) test=temp;
    }
    if (test < gtol) {
      return;
    }
    for (i=0;i<n;i++) dg(i)=g(i)-dg(i);
    for (i=0;i<n;i++) {
      hdg(i)=0.0;
      for (j=0;j<n;j++) hdg(i) += hessian(i,j)*dg(j);
    }
    fac=fae=sumdg=sumxi=0.0;
    for (i=0;i<n;i++) {
      fac += dg(i)*xi(i);
      fae += dg(i)*hdg(i);
      sumdg += SQR(dg(i));
      sumxi += SQR(xi(i));
    }
    if (fac*fac > EPS*sumdg*sumxi) {
      fac=1.0/fac;
      fad=1.0/fae;
      for (i=0;i<n;i++) dg(i)=fac*xi(i)-fad*hdg(i);
      for (i=0;i<n;i++) {
	for (j=0;j<n;j++) {
	  hessian(i,j) += fac*xi(i)*xi(j)
	    -fad*hdg(i)*hdg(j)+fae*dg(i)*dg(j);
	}
      }
    }
    for (i=0;i<n;i++) {
      xi(i)=0.0;
      for (j=0;j<n;j++) xi(i) -= hessian(i,j)*g(j);
    }
  }
  cout << "too many iterations in dfpmin" << endl;
}
#undef ITMAX
#undef EPS
#undef TOLX
#undef STPMX

#define ALF 1.0e-4
#define TOLX 1.0e-7

void lnsrch(int n, Vector &xold, double fold, Vector &g, Vector &p, Vector &x,
	    double *f, double stpmax, int *check, double (*func)(Vector &p))
{
  int i;
  double a,alam,alam2,alamin,b,disc,f2,fold2,rhs1,rhs2,slope,sum,temp,
    test,tmplam;

  *check=0;
  for (sum=0.0,i=0;i<n;i++) sum += p(i)*p(i);
  sum=sqrt(sum);
  if (sum > stpmax)
    for (i=0;i<n;i++) p(i) *= stpmax/sum;
  for (slope=0.0,i=0;i<n;i++)
    slope += g(i)*p(i);
  test=0.0;
  for (i=0;i<n;i++) {
    temp=fabs(p(i))/FMAX(fabs(xold(i)),1.0);
    if (temp > test) test=temp;
  }
  alamin=TOLX/test;
  alam=1.0;
  for (;;) {
    for (i=0;i<n;i++) x(i)=xold(i)+alam*p(i);
    *f=(*func)(x);
    if (alam < alamin) {
      for (i=0;i<n;i++) x(i)=xold(i);
      *check=1;
      return;
    } else if (*f <= fold+ALF*alam*slope) return;
    else {
      if (alam == 1.0)
	tmplam = -slope/(2.0*(*f-fold-slope));
      else {
	rhs1 = *f-fold-alam*slope;
	rhs2=f2-fold2-alam2*slope;
	a=(rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
	b=(-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
	if (a == 0.0) tmplam = -slope/(2.0*b);
	else {
	  disc=b*b-3.0*a*slope;
	  if (disc<0.0) cout << "Roundoff problem in lnsrch." << endl;
	  else tmplam=(-b+sqrt(disc))/(3.0*a);
	}
	if (tmplam>0.5*alam)
	  tmplam=0.5*alam;
      }
    }
    alam2=alam;
    f2 = *f;
    fold2=fold;
    alam=FMAX(tmplam,0.1*alam);
  }
}
#undef ALF
#undef TOLX


