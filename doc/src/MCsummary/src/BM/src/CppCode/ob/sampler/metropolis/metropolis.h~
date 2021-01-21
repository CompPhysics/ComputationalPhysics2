#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "sampler/sampler.h"

class Metropolis : public Sampler {
private:
    double m_psi;
    double m_step;
    double m_accepted;
    std::uniform_real_distribution<double> m_distributionStep;
    std::uniform_real_distribution<double> m_distributionTest;
    std::normal_distribution<double> m_distributionImportance;
public:
    Metropolis(int nSamples, int nCycles, double step, Hamiltonian &hamiltonian,
               NeuralQuantumState &nqs, Optimizer &optimizer,
               std::string filename, std::string blockFilename, int seed);
    void samplePositions(bool &accepted);
};

#endif // METROPOLIS_H
