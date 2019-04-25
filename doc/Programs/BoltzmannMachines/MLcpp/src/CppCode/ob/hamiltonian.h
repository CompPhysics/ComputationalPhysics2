#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "neuralquantumstate.h"

class Hamiltonian {
private:
    double m_omega;
    bool m_includeInteraction;

public:
    Hamiltonian(double omega, bool includeInteraction);
    double computeLocalEnergy(NeuralQuantumState &nqs);
    Eigen::VectorXd computeLocalEnergyGradientComponent(NeuralQuantumState &nqs);
    double interaction(Eigen::VectorXd x, int nx, int dim);
};

#endif // HAMILTONIAN_H
