#include "gibbs.h"

Gibbs::Gibbs(int nSamples, int nCycles, Hamiltonian &hamiltonian,
             NeuralQuantumState &nqs, Optimizer &optimizer,
             std::string filename,  std::string blockFilename, int seed) :
    Sampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename, blockFilename, seed) {

    m_distributionH = std::uniform_real_distribution<double>(0,1);
}


void Gibbs::samplePositions(bool &accepted) {
    // Set new hidden variables given positions, according to the logistic sigmoid function
    // (implemented by comparing the sigmoid probability to a uniform random variable)
    double z;
    for (int j=0; j<m_nqs.m_nh; j++) {
        z = m_nqs.m_b(j) + m_nqs.m_x.dot(m_nqs.m_w.col(j))/m_nqs.m_sig2;
        m_nqs.m_h(j) = m_distributionH(m_randomEngine) < logisticSigmoid(z);
    }

    // Set new positions (visibles) given hidden, according to normal distribution
    std::normal_distribution<double> distributionX;
    double xMean;
    for (int i=0; i<m_nqs.m_nx; i++) {
        xMean = m_nqs.m_a(i) + m_nqs.m_w.row(i)*m_nqs.m_h;
        distributionX = std::normal_distribution<double>(xMean, m_nqs.m_sig);
        m_nqs.m_x(i) = distributionX(m_randomEngine);
    }
}

double Gibbs::logisticSigmoid(double z) {
    return 1.0/(1+exp(-z));
}
