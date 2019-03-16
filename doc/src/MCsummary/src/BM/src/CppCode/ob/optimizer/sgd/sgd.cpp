#include "sgd.h"

Sgd::Sgd(double eta, double nPar) : Optimizer() {
    m_eta = eta;

    m_gradPrev.resize(nPar);
}

void Sgd::optimizeWeights(NeuralQuantumState &nqs, Eigen::VectorXd grad, int cycles) {
    // Compute new parameters
    for (int i=0; i<nqs.m_nx; i++) {
        //outfile << a(i) << " ";
        nqs.m_a(i) = nqs.m_a(i) - m_eta*grad(i);
    }
    for (int j=0; j<nqs.m_nh; j++) {
        //outfile << b(j) << " ";
        nqs.m_b(j) = nqs.m_b(j) - m_eta*grad(nqs.m_nx + j);
    }
    int k = nqs.m_nx + nqs.m_nh;
    for (int i=0; i<nqs.m_nx; i++) {
        for (int j=0; j<nqs.m_nh; j++) {
            //outfile << w(i,j) << " ";
            nqs.m_w(i,j) = nqs.m_w(i,j) - m_eta*grad(k);
            k++;
        }
    }
    //outfile << '\n';
}
