#ifndef METROPOLIS_H
#define METROPOLIS_H

#include <random>
#include "action.h"

class Metropolis
{
private:
    int N;
    int NCf;
    int NCor;
    int NTherm;
    double epsilon;
    double a;
    Action *S = nullptr;
    double (*gammaFunctional)(double * x, int n, int _N);

    void update(double *x, std::mt19937_64 &gen, std::uniform_real_distribution<double> &eps_distribution, std::uniform_real_distribution<double> &one_distribution, int &acceptanceCounter);
    void writeStatisticsToFile(double * dE, double * averagedGamma, double * averagedGammaSquared, int acceptanceCounter);
public:

    Metropolis(int new_N, int new_NCf, int new_NCor, int NTherm, double new_epsilon, double new_a);
    void runMetropolis();

    // Setters
    void setAction(Action *newS) { S = newS; }
    void setGammaFunctional(double (*newGammaFunctional)(double * x, int n, int _N)) { gammaFunctional = newGammaFunctional; }
    void setN(int new_N) { N = new_N; }
    void setNCf(int new_NCf) { NCf = new_NCf; }
    void setEpsilon(double new_epsilon) { epsilon = new_epsilon; }

    // Getters
    int getN() { return N; }
    int getNCf() { return NCf; }
    int getEpsilon() { return epsilon; }
};

#endif // METROPOLIS_H
