#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "neuralquantumstate.h"

class Optimizer {
public:
    Optimizer();
    virtual void optimizeWeights(NeuralQuantumState &nqs, Eigen::VectorXd grad, int cycles) = 0;
};

#endif // OPTIMIZER_H
