// Variational Monte Carlo for atoms with importance sampling, slater det
// Test case for 2-electron quantum dot, no classes using Mersenne-Twister RNG
#include "mpi.h"
#include <cmath>
#include <random>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "vectormatrixclass.h"

using namespace  std;
// output file as global variable
ofstream ofile;  
// the step length and its squared inverse for the second derivative 
//  Here we define global variables  used in various functions
//  These can be changed by using classes
int Dimension = 2; 
int NumberParticles  = 2;  //  we fix also the number of electrons to be 2

// declaration of functions 

// The Mc sampling for the variational Monte Carlo 
void  MonteCarloSampling(int, double &, double &, Vector &);

// The variational wave function
double  WaveFunction(Matrix &, Vector &);

// The local energy 
double  LocalEnergy(Matrix &, Vector &);

// The quantum force
void  QuantumForce(Matrix &, Matrix &, Vector &);


// inline function for single-particle wave function
inline double SPwavefunction(double r, double alpha) { 
   return exp(-alpha*r*0.5);
}

// inline function for derivative of single-particle wave function
inline double DerivativeSPwavefunction(double r, double alpha) { 
  return -r*alpha;
}

// function for absolute value of relative distance
double RelativeDistance(Matrix &r, int i, int j) { 
      double r_ij = 0;  
      for (int k = 0; k < Dimension; k++) { 
	r_ij += (r(i,k)-r(j,k))*(r(i,k)-r(j,k));
      }
      return sqrt(r_ij); 
}

// inline function for derivative of Jastrow factor
inline double JastrowDerivative(Matrix &r, double beta, int i, int j, int k){
  return (r(i,k)-r(j,k))/(RelativeDistance(r, i, j)*pow(1.0+beta*RelativeDistance(r, i, j),2));
}

// function for square of position of single particle
double singleparticle_pos2(Matrix &r, int i) { 
    double r_single_particle = 0;
    for (int j = 0; j < Dimension; j++) { 
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

  //  MPI initializations
  int NumberProcesses, MyRank, NumberMCsamples;
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &NumberProcesses);
  MPI_Comm_rank (MPI_COMM_WORLD, &MyRank);
  double StartTime = MPI_Wtime();
  if (MyRank == 0 && argc <= 1) {
    cout << "Bad Usage: " << argv[0] << 
      " Read also output file on same line and number of Monte Carlo cycles" << endl;
  }
  // Read filename and number of Monte Carlo cycles from the command line
  if (MyRank == 0 && argc > 2) {
    string filename = argv[1]; // first command line argument after name of program
    NumberMCsamples  = atoi(argv[2]);
    string fileout = filename;
    string argument = to_string(NumberMCsamples);
    // Final filename as filename+NumberMCsamples
    fileout.append(argument);
    ofile.open(fileout);
  }
  // broadcast the number of  Monte Carlo samples
  MPI_Bcast (&NumberMCsamples, 1, MPI_INT, 0, MPI_COMM_WORLD);
  // Two variational parameters only
  Vector VariationalParameters(2);
  int TotalNumberMCsamples = NumberMCsamples*NumberProcesses; 
  // Loop over variational parameters
  for (double alpha = 0.5; alpha <= 1.5; alpha +=0.1){
    for (double beta = 0.1; beta <= 0.5; beta +=0.05){
      VariationalParameters(0) = alpha;  // value of alpha
      VariationalParameters(1) = beta;  // value of beta
      //  Do the mc sampling  and accumulate data with MPI_Reduce
      double TotalEnergy, TotalEnergySquared, LocalProcessEnergy, LocalProcessEnergy2;
      LocalProcessEnergy = LocalProcessEnergy2 = 0.0;
      MonteCarloSampling(NumberMCsamples, LocalProcessEnergy, LocalProcessEnergy2, VariationalParameters);
      //  Collect data in total averages
      MPI_Reduce(&LocalProcessEnergy, &TotalEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&LocalProcessEnergy2, &TotalEnergySquared, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      // Print out results  in case of Master node, set to MyRank = 0
      if ( MyRank == 0) {
	double Energy = TotalEnergy/( (double)NumberProcesses);
	double Variance = TotalEnergySquared/( (double)NumberProcesses)-Energy*Energy;
	double StandardDeviation = sqrt(Variance/((double)TotalNumberMCsamples)); // over optimistic error
	ofile << setiosflags(ios::showpoint | ios::uppercase);
	ofile << setw(15) << setprecision(8) << VariationalParameters(0);
	ofile << setw(15) << setprecision(8) << VariationalParameters(1);
	ofile << setw(15) << setprecision(8) << Energy;
	ofile << setw(15) << setprecision(8) << Variance;
	ofile << setw(15) << setprecision(8) << StandardDeviation << endl;
      }
    }
  }
  double EndTime = MPI_Wtime();
  double TotalTime = EndTime-StartTime;
  if ( MyRank == 0 )  cout << "Time = " <<  TotalTime  << " on number of processors: "  << NumberProcesses  << endl;
  if (MyRank == 0)  ofile.close();  // close output file
  // End MPI
  MPI_Finalize ();  
  return 0;
}  //  end of main function


// Monte Carlo sampling with the Metropolis algorithm  

void MonteCarloSampling(int NumberMCsamples, double &cumulative_e, double &cumulative_e2, Vector &VariationalParameters)
{

 // Initialize the seed and call the Mersienne algo
  std::random_device rd;
  std::mt19937_64 gen(rd());
  // Set up the uniform distribution for x \in [[0, 1]
  std::uniform_real_distribution<double> UniformNumberGenerator(0.0,1.0);
  std::normal_distribution<double> Normaldistribution(0.0,1.0);
  // diffusion constant from Schroedinger equation
  double D = 0.5; 
  double timestep = 0.05;  //  we fix the time step  for the gaussian deviate
  // allocate matrices which contain the position of the particles  
  Matrix OldPosition( NumberParticles, Dimension), NewPosition( NumberParticles, Dimension);
  Matrix OldQuantumForce(NumberParticles, Dimension), NewQuantumForce(NumberParticles, Dimension);
  double Energy = 0.0; double EnergySquared = 0.0; double DeltaE = 0.0;
  //  initial trial positions
  for (int i = 0; i < NumberParticles; i++) { 
    for (int j = 0; j < Dimension; j++) {
      OldPosition(i,j) = Normaldistribution(gen)*sqrt(timestep);
    }
  }
  double OldWaveFunction = WaveFunction(OldPosition, VariationalParameters);
  QuantumForce(OldPosition, OldQuantumForce, VariationalParameters);
  // loop over monte carlo cycles 
  for (int cycles = 1; cycles <= NumberMCsamples; cycles++){ 
    // new position 
    for (int i = 0; i < NumberParticles; i++) { 
      for (int j = 0; j < Dimension; j++) {
	// gaussian deviate to compute new positions using a given timestep
	NewPosition(i,j) = OldPosition(i,j) + Normaldistribution(gen)*sqrt(timestep)+OldQuantumForce(i,j)*timestep*D;
	//	NewPosition(i,j) = OldPosition(i,j) + gaussian_deviate(&idum)*sqrt(timestep)+OldQuantumForce(i,j)*timestep*D;
      }  
      //  for the other particles we need to set the position to the old position since
      //  we move only one particle at the time
      for (int k = 0; k < NumberParticles; k++) {
	if ( k != i) {
	  for (int j = 0; j < Dimension; j++) {
	    NewPosition(k,j) = OldPosition(k,j);
	  }
	} 
      }
      double NewWaveFunction = WaveFunction(NewPosition, VariationalParameters); 
      QuantumForce(NewPosition, NewQuantumForce, VariationalParameters);
      //  we compute the log of the ratio of the greens functions to be used in the 
      //  Metropolis-Hastings algorithm
      double GreensFunction = 0.0;            
      for (int j = 0; j < Dimension; j++) {
	GreensFunction += 0.5*(OldQuantumForce(i,j)+NewQuantumForce(i,j))*
	  (D*timestep*0.5*(OldQuantumForce(i,j)-NewQuantumForce(i,j))-NewPosition(i,j)+OldPosition(i,j));
      }
      GreensFunction = exp(GreensFunction);
      // The Metropolis test is performed by moving one particle at the time
      if(UniformNumberGenerator(gen) <= GreensFunction*NewWaveFunction*NewWaveFunction/OldWaveFunction/OldWaveFunction ) { 
	for (int  j = 0; j < Dimension; j++) {
	  OldPosition(i,j) = NewPosition(i,j);
	  OldQuantumForce(i,j) = NewQuantumForce(i,j);
	}
	OldWaveFunction = NewWaveFunction;
      }
    }  //  end of loop over particles
    // compute local energy  
    double DeltaE = LocalEnergy(OldPosition, VariationalParameters);
    // update energies
    Energy += DeltaE;
    EnergySquared += DeltaE*DeltaE;
  }   // end of loop over MC trials   
  // update the energy average and its squared 
  cumulative_e = Energy/NumberMCsamples;
  cumulative_e2 = EnergySquared/NumberMCsamples;
}   // end MonteCarloSampling function  


// Function to compute the squared wave function and the quantum force

double  WaveFunction(Matrix &r, Vector &VariationalParameters)
{
  double wf = 0.0;
  // full Slater determinant for two particles, replace with Slater det for more particles 
  wf  = SPwavefunction(singleparticle_pos2(r, 0), VariationalParameters(0))*SPwavefunction(singleparticle_pos2(r, 1),VariationalParameters(0));
  // contribution from Jastrow factor
  for (int i = 0; i < NumberParticles-1; i++) { 
    for (int j = i+1; j < NumberParticles; j++) {
      wf *= exp(RelativeDistance(r, i, j)/((1.0+VariationalParameters(1)*RelativeDistance(r, i, j))));
    }
  }
  return wf;
}

// Function to calculate the local energy without numerical derivation of kinetic energy

double  LocalEnergy(Matrix &r, Vector &VariationalParameters)
{

  // compute the kinetic and potential energy from the single-particle part
  // for a many-electron system this has to be replaced by a Slater determinant
  // The absolute value of the interparticle length
  Matrix length( NumberParticles, NumberParticles);
  // Set up interparticle distance
  for (int i = 0; i < NumberParticles-1; i++) { 
    for(int j = i+1; j < NumberParticles; j++){
      length(i,j) = RelativeDistance(r, i, j);
      length(j,i) =  length(i,j);
    }
  }
  double KineticEnergy = 0.0;
  // Set up kinetic energy from Slater and Jastrow terms
  for (int i = 0; i < NumberParticles; i++) { 
    for (int k = 0; k < Dimension; k++) {
      double sum1 = 0.0; 
      for(int j = 0; j < NumberParticles; j++){
	if ( j != i) {
	  sum1 += JastrowDerivative(r, VariationalParameters(1), i, j, k);
	}
      }
      KineticEnergy += (sum1+DerivativeSPwavefunction(r(i,k),VariationalParameters(0)))*(sum1+DerivativeSPwavefunction(r(i,k),VariationalParameters(0)));
    }
  }
  KineticEnergy += -2*VariationalParameters(0)*NumberParticles;
  for (int i = 0; i < NumberParticles-1; i++) {
      for (int j = i+1; j < NumberParticles; j++) {
        KineticEnergy += 2.0/(pow(1.0 + VariationalParameters(1)*length(i,j),2))*(1.0/length(i,j)-2*VariationalParameters(1)/(1+VariationalParameters(1)*length(i,j)) );
      }
  }
  KineticEnergy *= -0.5;
  // Set up potential energy, external potential + eventual electron-electron repulsion
  double PotentialEnergy = 0;
  for (int i = 0; i < NumberParticles; i++) { 
    double DistanceSquared = singleparticle_pos2(r, i);
    PotentialEnergy += 0.5*DistanceSquared;  // sp energy HO part, note it has the oscillator frequency set to 1!
  }
  // Add the electron-electron repulsion
  for (int i = 0; i < NumberParticles-1; i++) { 
    for (int j = i+1; j < NumberParticles; j++) {
      PotentialEnergy += 1.0/length(i,j);          
    }
  }
  double LocalE = KineticEnergy+PotentialEnergy;
  return LocalE;
}

// Compute the analytical expression for the quantum force
void  QuantumForce(Matrix &r, Matrix &qforce, Vector &VariationalParameters)
{
  // compute the first derivative 
  for (int i = 0; i < NumberParticles; i++) {
    for (int k = 0; k < Dimension; k++) {
      // single-particle part, replace with Slater det for larger systems
      double sppart = DerivativeSPwavefunction(r(i,k),VariationalParameters(0));
      //  Jastrow factor contribution
      double Jsum = 0.0;
      for (int j = 0; j < NumberParticles; j++) {
	if ( j != i) {
	  Jsum += JastrowDerivative(r, VariationalParameters(1), i, j, k);
	}
      }
      qforce(i,k) = 2.0*(Jsum+sppart);
    }
  }
} // end of QuantumForce function


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






