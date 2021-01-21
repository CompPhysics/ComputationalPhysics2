#ifndef ASGD_H
#define ASGD_H

#include "optimizer/optimizer.h"

class Asgd : public Optimizer {
private:
    // Parameters to Asgd
    double m_A;
    double m_a;
    double m_omega;
    double m_fmax;
    double m_fmin;
    double m_t;


    // Variables that are updated, then used in the following iteration/
    // call to uptimizeWeights()
    Eigen::VectorXd m_gradPrev;
    double m_tprev;
public:
    Asgd(double a, double A, double asgdOmega, double fmax, double fmin,
               double t0, double t1, int nPar);
    Asgd(double a, double A, double nPar);
    void optimizeWeights(NeuralQuantumState &nqs, Eigen::VectorXd grad, int cycles);
};

#endif // ASGD_H
