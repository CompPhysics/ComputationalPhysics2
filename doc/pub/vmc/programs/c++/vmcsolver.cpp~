#include "vmcsolver.h"
#include "lib.h"

#include <armadillo>
#include <iostream>

using namespace arma;
using namespace std;

VMCSolver::VMCSolver() :
    nDimensions(3),
    charge(2),
    nParticles(2),
    h(0.001),
    h2(1000000),
    idum(-1),
    alpha(0.5*charge),
    nCycles(1000000),
    timestep(0.05),
    D(0.5)
{
}

void VMCSolver::runMonteCarloIntegration()
{
  rOld = zeros<mat>(nParticles, nDimensions);
  rNew = zeros<mat>(nParticles, nDimensions);
  QForceOld = zeros<mat>(nParticles, nDimensions);
  QForceNew = zeros<mat>(nParticles, nDimensions);

  double waveFunctionOld = 0;
  double waveFunctionNew = 0;

  double energySum = 0;
  double energySquaredSum = 0;

  double deltaE;

  // initial trial positions
  for(int i = 0; i < nParticles; i++) {
    for(int j = 0; j < nDimensions; j++) {
      rOld(i,j) = GaussianDeviate(&idum)*sqrt(timestep);
    }
  }
  rNew = rOld;

  // loop over Monte Carlo cycles
  for(int cycle = 0; cycle < nCycles; cycle++) {

    // Store the current value of the wave function
    waveFunctionOld = waveFunction(rOld);
    QuantumForce(rOld, QForceOld); QForceOld = QForceOld*h/waveFunctionOld;
    // New position to test
    for(int i = 0; i < nParticles; i++) {
      for(int j = 0; j < nDimensions; j++) {
	rNew(i,j) = rOld(i,j) + GaussianDeviate(&idum)*sqrt(timestep)+QForceOld(i,j)*timestep*D;
      }
      //  for the other particles we need to set the position to the old position since
      //  we move only one particle at the time
      for (int k = 0; k < nParticles; k++) {
	if ( k != i) {
	  for (int j=0; j < nDimensions; j++) {
	    rNew(k,j) = rOld(k,j);
	  }
	} 
      }
      // Recalculate the value of the wave function and the quantum force
      waveFunctionNew = waveFunction(rNew);
      QuantumForce(rNew,QForceNew); QForceNew*h/waveFunctionNew;
      //  we compute the log of the ratio of the greens functions to be used in the 
      //  Metropolis-Hastings algorithm
      GreensFunction = 0.0;            
      for (int j=0; j < nDimensions; j++) {
	GreensFunction += 0.5*(QForceOld(i,j)+QForceNew(i,j))*
	  (D*timestep*0.5*(QForceOld(i,j)-QForceNew(i,j))-rNew(i,j)+rOld(i,j));
      }
      GreensFunction = exp(GreensFunction);

      // The Metropolis test is performed by moving one particle at the time
      if(ran2(&idum) <= GreensFunction*(waveFunctionNew*waveFunctionNew) / (waveFunctionOld*waveFunctionOld)) {
	for(int j = 0; j < nDimensions; j++) {
	  rOld(i,j) = rNew(i,j);
	  QForceOld(i,j) = QForceNew(i,j);
	  waveFunctionOld = waveFunctionNew;
	}
      } else {
	for(int j = 0; j < nDimensions; j++) {
	  rNew(i,j) = rOld(i,j);
	  QForceNew(i,j) = QForceOld(i,j);
	}
      }
      // update energies
      deltaE = localEnergy(rNew);
      energySum += deltaE;
      energySquaredSum += deltaE*deltaE;
    }
  }
  double energy = energySum/(nCycles * nParticles);
  double energySquared = energySquaredSum/(nCycles * nParticles);
  cout << "Energy: " << energy << " Energy (squared sum): " << energySquared << endl;
}

double VMCSolver::localEnergy(const mat &r)
{
    mat rPlus = zeros<mat>(nParticles, nDimensions);
    mat rMinus = zeros<mat>(nParticles, nDimensions);

    rPlus = rMinus = r;

    double waveFunctionMinus = 0;
    double waveFunctionPlus = 0;

    double waveFunctionCurrent = waveFunction(r);

    // Kinetic energy

    double kineticEnergy = 0;
    for(int i = 0; i < nParticles; i++) {
        for(int j = 0; j < nDimensions; j++) {
            rPlus(i,j) += h;
            rMinus(i,j) -= h;
            waveFunctionMinus = waveFunction(rMinus);
            waveFunctionPlus = waveFunction(rPlus);
            kineticEnergy -= (waveFunctionMinus + waveFunctionPlus - 2 * waveFunctionCurrent);
            rPlus(i,j) = r(i,j);
            rMinus(i,j) = r(i,j);
        }
    }
    kineticEnergy = 0.5 * h2 * kineticEnergy / waveFunctionCurrent;

    // Potential energy
    double potentialEnergy = 0;
    double rSingleParticle = 0;
    for(int i = 0; i < nParticles; i++) {
        rSingleParticle = 0;
        for(int j = 0; j < nDimensions; j++) {
            rSingleParticle += r(i,j)*r(i,j);
        }
        potentialEnergy -= charge / sqrt(rSingleParticle);
    }
    // Contribution from electron-electron potential
    double r12 = 0;
    for(int i = 0; i < nParticles; i++) {
        for(int j = i + 1; j < nParticles; j++) {
            r12 = 0;
            for(int k = 0; k < nDimensions; k++) {
                r12 += (r(i,k) - r(j,k)) * (r(i,k) - r(j,k));
            }
            potentialEnergy += 1 / sqrt(r12);
        }
    }

    return kineticEnergy + potentialEnergy;
}

double VMCSolver::waveFunction(const mat &r)
{
    double argument = 0;
    for(int i = 0; i < nParticles; i++) {
        double rSingleParticle = 0;
        for(int j = 0; j < nDimensions; j++) {
            rSingleParticle += r(i,j) * r(i,j);
        }
        argument += sqrt(rSingleParticle);
    }
    return exp(-argument * alpha);
}



double VMCSolver::QuantumForce(const mat &r, mat QForce)
{
    mat rPlus = zeros<mat>(nParticles, nDimensions);
    mat rMinus = zeros<mat>(nParticles, nDimensions);

    rPlus = rMinus = r;

    double waveFunctionMinus = 0;
    double waveFunctionPlus = 0;

    double waveFunctionCurrent = waveFunction(r);

    // Kinetic energy

    double kineticEnergy = 0;
    for(int i = 0; i < nParticles; i++) {
        for(int j = 0; j < nDimensions; j++) {
            rPlus(i,j) += h;
            rMinus(i,j) -= h;
            waveFunctionMinus = waveFunction(rMinus);
            waveFunctionPlus = waveFunction(rPlus);
            QForce(i,j) =  (waveFunctionPlus-waveFunctionMinus);
            rPlus(i,j) = r(i,j);
            rMinus(i,j) = r(i,j);
        }
    }
}
