#include <random>   // For Mersenne-Twister19937
#include <ctime>    // For random seed
#include <cmath>    // For exp()
#include <fstream>
#include <iostream>
#include "metropolis.h"
#include "action.h"

using std::cout;
using std::endl;
void printArray(double *x, int N);

Metropolis::Metropolis(int new_N, int new_NCf, int new_NCor, int new_Therm, double new_epsilon, double new_a)
{
    N = new_N;
    NCf = new_NCf;
    NCor = new_NCor;
    NTherm = new_Therm;
    epsilon = new_epsilon;
    a = new_a;
}

void Metropolis::update(double *x,
                        std::mt19937_64 &gen,
                        std::uniform_real_distribution<double> &eps_distribution,
                        std::uniform_real_distribution<double> &one_distribution,
                        int &acceptanceCounter)
{
    for (int i = 0; i < N; i++)
    {
        double x_prev = x[i];
        double oldS = S->getAction(x, i);
        x[i] += eps_distribution(gen); // setting a new possible x-position to test for
        double deltaS = S->getAction(x, i) - oldS;

        if ((deltaS > 0) && (exp(-deltaS) < one_distribution(gen)))
        {
            x[i] = x_prev;
        }
        else
        {
            acceptanceCounter++;
        }
    }
}

void Metropolis::runMetropolis()
{
    // Setting up random generators
    std::mt19937_64 generator(std::time(nullptr)); // Starting up the Mersenne-Twister19937 function
    std::uniform_real_distribution<double> eps_distribution(-epsilon, epsilon);
    std::uniform_real_distribution<double> one_distribution(0,1);

    // Setting up array for averaged Gamma-values, variance and Metropolis acceptance counter
    int acceptanceCounter = 0;
    double * averagedGamma = new double[N];
    double * averagedGammaSquared = new double[N];
    double * deltaE = new double[N];
    for (int i = 0; i < N; i++)
    {
        averagedGamma[i] = 0;
        averagedGammaSquared[i] = 0;
        deltaE[i] = 0;
    }

    // Setting up array for Gamma-functional values
    double ** Gamma = new double*[NCf];
    for (int i = 0; i < NCf; i++) { Gamma[i] = new double[N]; }
    for (int i = 0; i < NCf; i++) { for (int j = 0; j < N; j++) { Gamma[i][j] = 0; } } // Setting matrix elements to zero

    // Setting up array
    double * x = new double[N]; // Only need one array, as it will always be updated. Note, it is 1D
    for (int i = 0; i < N; i++) { x[i] = 0; }

    // Running thermalization
    for (int i = 0; i < NTherm * NCor; i++)
    {
        update(x, generator, eps_distribution, one_distribution, acceptanceCounter);
    }

    // Setting the Metropolis acceptance counter to 0 in order not to count the thermalization
    acceptanceCounter = 0;

    // Main part of algorithm
    for (int alpha = 0; alpha < NCf; alpha++)
    {
        for (int i = 0; i < NCor; i++) // Updating NCor times before updating the Gamma function
        {
            update(x, generator, eps_distribution, one_distribution, acceptanceCounter);
        }
        for (int n = 0; n < N; n++)
        {
            Gamma[alpha][n] = gammaFunctional(x,n,N);
        }
    }

    // Performing an average over the Monte Carlo obtained values
    for (int n = 0; n < N; n++)
    {
        for (int alpha = 0; alpha < NCf; alpha++)
        {
            averagedGamma[n] += Gamma[alpha][n];
            averagedGammaSquared[n] += Gamma[alpha][n]*Gamma[alpha][n];
        }
        averagedGamma[n] /= double(NCf);
        averagedGammaSquared[n] /= double(NCf);
    }

    // Getting change in energy
    for (int n = 0; n < N-1; n++)
    {
        deltaE[n] = log(averagedGamma[n]/averagedGamma[n+1])/a;
    }
    deltaE[N-1] = log(averagedGamma[N-1]/averagedGamma[0])/a; // Ensuring that the last energy also gets counted

    // Printing the energies in the calculation
    for (int n = 0; n < N; n++)
    {
        cout << deltaE[n] << endl;
    }

    // Printing deltaE, variance, standard deviation to file
    writeStatisticsToFile(deltaE, averagedGamma, averagedGammaSquared, acceptanceCounter);

    printf("Acceptancerate: %f \n", double(acceptanceCounter)/double(NCf*NCor*N));

    // De-allocating arrays
    for (int i = 0; i < NCf; i++) { delete [] Gamma[i]; }
    delete [] x;
    delete [] Gamma;
    delete [] averagedGamma;
    delete [] averagedGammaSquared;
    delete [] deltaE;
}

void Metropolis::writeStatisticsToFile(double * dE, double * averagedGamma, double * averagedGammaSquared, int acceptanceCounter)
{
    /*
     * Writes statistics to file about:
     * acceptanceCounter:   the number of accepted configurations
     * NCor:                number of times between each sampling of the functional
     * NCf:                 number of paths we are looking at
     * t=n*a:               points on lattice
     * dE:                  energy for a given point on lattice
     * dE error:            the error in the dE measurment
     * variance:            Var(G)
     * standardDeviation:   std(G)
     */
    double * variance = new double[N];
    double * standardDeviation = new double[N];
    for (int i = 0; i < N; i++)
    {
        variance[i] = 0;
        standardDeviation[i] = 0;
    }

    // Calculating variance and standard deviation of G
    for (int i = 0; i < N; i++)
    {
        variance[i] = (averagedGammaSquared[i] - averagedGamma[i]*averagedGamma[i])/NCf;
        standardDeviation[i] = sqrt(variance[i]);
    }

    // Initializing file and writing to file
    std::ofstream file;
    file.open("statistics.txt");
    file << "acceptanceCounter " << double(acceptanceCounter)/double(N*NCf*NCor) << endl;
    file << "NCor " << NCor << endl;
    file << "NCf " << NCf << endl;
    file << "NTherm " << NTherm << endl;
    for (int n = 0; n < N; n++)
    {
        file << n*a << " "
             << dE[n] << " "
             << sqrt((standardDeviation[n]/averagedGamma[n])*(standardDeviation[n]/averagedGamma[n]) + (standardDeviation[(n+1)%N]/averagedGamma[(n+1)%N])*(standardDeviation[(n+1)%N]/averagedGamma[(n+1)%N]))/a << " "
             << variance[n] << " "
             << standardDeviation[n] << endl;
    }
    file.close();
    cout << "statistics.txt written" << endl;

    delete [] variance;
    delete [] standardDeviation;
}

void printArray(double *x, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%.2f \n", x[i]);
    }
}
