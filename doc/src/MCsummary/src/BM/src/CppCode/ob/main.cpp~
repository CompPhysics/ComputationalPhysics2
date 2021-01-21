#include <iostream>
#include "sampler/metropolis/metropolis.h"
#include "sampler/gibbs/gibbs.h"

using namespace std;

int main() {
    // filenames
    string filename = "/Users/Vilde/Documents/masters/NQS_paper/tryHOrbm/RBMoutput.txt";
    string blockFilename = "/Users/Vilde/Documents/masters/NQS_paper/tryHOrbm/blocking.txt";

    // Nqs parameters
    int nx = 4;                          // Number which represents particles*dimensions.
    int nh = 2;                          // Number of hidden units.
    int dim = 2;                         // Number of spatial dimensions
    double sigma = 1.0;                  // Normal distribution visibles
    bool gaussianInitialization = true; // Weights & biases (a,b,w) initialized uniformly or gaussian

    // Sampler parameters
    int nCycles = 100;                   // Number of optimization iterations
    int nSamples = 100000;           // Number of samples in each iteration
    random_device rd;                    // Seed
    // Metropolis
    double step = 0.45;

    // Hamiltonian parameters
    double omega = 1.0;
    bool includeInteraction = true;      // Include interaction or not

    // Optimizer parameters (choose either stochastic gradient descent (SGD) or adaptive SGD (ASGD))
    int nPar = nx + nh + nx*nh;
    // SGD parameters
    double eta = 0.01;                   // must be >0. SGD learning rate (lr)
    /*
    // ASGD parameters. lr: gamma_i=a/(A+t_i) where t[i]=max(0, t[i-1]+f(-grad[i]*grad[i-1]))
    double a = 0.01;                     // must be >0. Proportional to the lr
    double A = 20.0;                     // must be >= 1. Inverse prop to the lr. (a/A) defines the max lr.
    // ASGD optional: parameters to the function f
    double asgdOmega;                    // must be >0. As omega->0, f-> step function.
    double fmax;                         // must be >0
    double fmin;                         // must be <0
    // ASGD optional: initial conditions
    double t0;                           // Suggested choices are t0=t1=A=20 (default)
    double t1;                           // or t0=t1=0
    */



    // Create objects for the sampler:
    Hamiltonian hamiltonian(omega, includeInteraction);
    NeuralQuantumState nqs(nh, nx, dim, sigma, gaussianInitialization);
    Sgd optimizer(eta, nPar);

    // Create the sampler:
    Metropolis metropolisSampler(nSamples, nCycles, step, hamiltonian, nqs, optimizer, filename,
                                 blockFilename, rd());
    //Gibbs gibbsSampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename, blockFilename);

    // Run
    metropolisSampler.runOptimizationSampling();
    //gibbsSampler.runOptimizationSampling();


    return 0;
}
