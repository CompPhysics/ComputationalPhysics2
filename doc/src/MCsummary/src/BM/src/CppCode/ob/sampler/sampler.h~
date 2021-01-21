#ifndef SAMPLER_H
#define SAMPLER_H

#include "hamiltonian.h"
#include "optimizer/sgd/sgd.h"
#include "optimizer/asgd/asgd.h"
#include <random>
#include <fstream>

class Sampler {
protected:
    int m_nSamples;
    int m_nCycles;
    std::ofstream m_outfile;
    std::ofstream m_blockOutfile;
    std::string m_blockFilename;
    std::mt19937_64 m_randomEngine;


public:
    Hamiltonian m_hamiltonian;
    NeuralQuantumState m_nqs;
    Optimizer &m_optimizer; // I put & here bc not allowed to instanciate an abstract class. Bad practice??
    // See more: https://stackoverflow.com/questions/12387239/reference-member-variables-as-class-members
    Sampler(int nSamples, int nCycles, Hamiltonian &hamiltonian,
            NeuralQuantumState &nqs, Optimizer &optimizer,
            std::string filename, std::string blockFilename, int seed);
    void runOptimizationSampling();
    virtual void samplePositions(bool &accepted) = 0;
};

#endif // SAMPLER_H
