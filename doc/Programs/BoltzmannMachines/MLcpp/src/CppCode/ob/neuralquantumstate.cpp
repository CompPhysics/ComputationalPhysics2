#include "neuralquantumstate.h"
#include <random>

NeuralQuantumState::NeuralQuantumState(int nh, int nx, int dim, double sigma, bool gaussianInitialization) {
    std::random_device rd;
    m_randomEngine = std::mt19937_64(rd());
    setup(nh, nx, dim, sigma, gaussianInitialization);
}

NeuralQuantumState::NeuralQuantumState(int nh, int nx, int dim, double sigma, bool gaussianInitialization, int seed) {
    m_randomEngine = std::mt19937_64(seed);
    setup(nh, nx, dim, sigma, gaussianInitialization);
}

void NeuralQuantumState::setup(int nh, int nx, int dim, double sigma, bool gaussianInitialization) {
    m_nx = nx;
    m_nh = nh;
    m_dim  = dim;
    m_sig = sigma;
    m_sig2 = sigma*sigma;
    m_x.resize(m_nx); // positions/visibles
    m_h.resize(m_nh); // hidden
    m_a.resize(m_nx); // visible bias
    m_b.resize(m_nh); // hidden bias
    m_w.resize(m_nx, m_nh); // weights
    if (gaussianInitialization) {
        setupWeights();
    } else {
        m_x = Eigen::VectorXd::Random(nx);
        m_a = Eigen::VectorXd::Random(nx);
        m_b = Eigen::VectorXd::Random(nh);
        m_w = Eigen::MatrixXd::Random(nx, nh);
    }

    setupPositions();
}

void NeuralQuantumState::setupWeights() {
    float sigma_initRBM = 0.001;
    std::normal_distribution<double> distribution_initRBM(0,sigma_initRBM);
    for (int i=0; i<m_nx; i++){
        m_a(i) = distribution_initRBM(m_randomEngine);
        //outfile << a(i) << " ";
    }
    for (int i=0; i<m_nh; i++){
        m_b(i) = distribution_initRBM(m_randomEngine);
        //outfile << b(i) << " ";
    }
    for (int i=0; i<m_nx; i++){
        for (int j=0; j<m_nh; j++){
            m_w(i,j) = distribution_initRBM(m_randomEngine);
            //outfile << w(i,j) << " ";
        }
    }
    //outfile << '\n';
}

void NeuralQuantumState::setupPositions() {
    std::uniform_real_distribution<double> distribution_initX(-0.5,0.5);
    for(int i=0; i<m_nx; i++){
        m_x(i)=distribution_initX(m_randomEngine);
    }
}

double NeuralQuantumState::computePsi() {
    m_psiFactor1 = 0.0;
    for (int i=0; i<m_nx; i++) {
        m_psiFactor1 += (m_x(i) - m_a(i))*(m_x(i) - m_a(i));
    }
    m_psiFactor2 = 1.0;
    m_Q = m_b + (((1.0/m_sig2)*m_x).transpose()*m_w).transpose();
    for (int j=0; j<m_nh; j++) {
        m_psiFactor2 *= (1 + exp(m_Q(j)));
    }
    m_psiFactor1 = exp(-m_psiFactor1/(2.0*m_sig2));
    return m_psiFactor1*m_psiFactor2;
}

double NeuralQuantumState::computePsi(Eigen::VectorXd x) {
    m_psiFactor1 = 0.0;
    for (int i=0; i<m_nx; i++) {
        m_psiFactor1 += (x(i) - m_a(i))*(x(i) - m_a(i));
    }
    m_psiFactor2 = 1.0;
    m_Q = m_b + (((1.0/m_sig2)*x).transpose()*m_w).transpose();
    for (int j=0; j<m_nh; j++) {
        m_psiFactor2 *= (1 + exp(m_Q(j)));
    }
    m_psiFactor1 = exp(-m_psiFactor1/(2.0*m_sig2));
    return m_psiFactor1*m_psiFactor2;
}
