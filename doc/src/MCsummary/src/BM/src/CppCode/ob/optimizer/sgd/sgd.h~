#ifndef SGD_H
#define SGD_H

#include "optimizer/optimizer.h"

class Sgd : public Optimizer {
private:
    // Parameters for Sgd
    double m_eta;

    // Variables that are updated, then used in the following iteration/
    // call to uptimizeWeights()
    double m_asgdXprev;
    Eigen::VectorXd m_gradPrev;
    double m_tprev;
public:
    Sgd(double eta, double nPar);
    void optimizeWeights(NeuralQuantumState &nqs, Eigen::VectorXd grad, int cycles);
};

#endif // SGD_H
