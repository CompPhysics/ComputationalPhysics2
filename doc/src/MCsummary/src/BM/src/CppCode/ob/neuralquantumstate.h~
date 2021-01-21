#ifndef NEURALQUANTUMSTATE_H
#define NEURALQUANTUMSTATE_H

#include <Eigen/Dense>
#include <random>

class NeuralQuantumState {
private:
    double m_psiFactor1;
    double m_psiFactor2;
    Eigen::VectorXd m_Q;
    std::mt19937_64 m_randomEngine; // For the distributions


    void setup(int nh, int nx, int dim, double sigma, bool gaussianInitialization);
    void setupWeights();
    void setupPositions();

public:
    int m_nx;
    int m_nh;
    int m_dim;
    double m_sig;
    double m_sig2;
    Eigen::VectorXd m_x;
    Eigen::VectorXd m_h;
    Eigen::VectorXd m_a;
    Eigen::VectorXd m_b;
    Eigen::MatrixXd m_w;

    NeuralQuantumState(int nh, int nx, int dim, double sigma, bool gaussianInitialization);
    NeuralQuantumState(int nh, int nx, int dim, double sigma, bool gaussianInitialization, int seed);
    double computePsi();
    double computePsi(Eigen::VectorXd x); // Needed for Sampler Metropolis method
};

#endif // NEURALQUANTUMSTATE_H
