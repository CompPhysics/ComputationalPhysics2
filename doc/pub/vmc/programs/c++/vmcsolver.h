#ifndef VMCSOLVER_H
#define VMCSOLVER_H

#include <armadillo>

using namespace arma;

class VMCSolver
{
public:
    VMCSolver();

    void runMonteCarloIntegration();

private:
    double waveFunction(const mat &r);
    double localEnergy(const mat &r);
    double QuantumForce(const mat &r, mat F);

    int nDimensions;
    int charge;
    double stepLength;
    int nParticles;

    double h;
    double h2;
// diffusion constant from Schroedinger equation
    double D;
//  we fix the time step  for the gaussian deviate
    double timestep;  
    long idum;

    double alpha;
    double GreensFunction;
    int nCycles;

    mat rOld;
    mat rNew;
    mat QForceOld;
    mat QForceNew;
};

#endif // VMCSOLVER_H
