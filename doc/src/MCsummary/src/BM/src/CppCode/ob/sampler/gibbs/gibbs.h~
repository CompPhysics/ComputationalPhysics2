#ifndef GIBBS_H
#define GIBBS_H

#include "sampler/sampler.h"

class Gibbs : public Sampler {
private:
    std::uniform_real_distribution<double> m_distributionH;

public:
    Gibbs(int nSamples, int nCycles, Hamiltonian &hamiltonian,
          NeuralQuantumState &nqs, Optimizer &optimizer,
          std::string filename, std::string blockFilename, int seed);
    void samplePositions(bool &accepted);
    double logisticSigmoid(double z);
};

#endif // GIBBS_H
