#include "wavefunction.h"
#include <cmath>
#include <iostream>
#include "eigen3/Eigen/Dense"

using namespace std;
using namespace Eigen;

double rij(VectorXd X, int D) {
    double Ep = 0;              // Sum 1/rij
    int P = X.size()/D;
    for(int i=0; i<P; i++) {
        for(int j=0; j<i; j++) {
            double dist = 0;
            for(int d=0; d<D; d++) {
                dist += (X(D*i+d)-X(D*j+d))*(X(D*i+d)-X(D*j+d));
            }
            Ep += 1/sqrt(dist);
        }
    }
    return Ep;
}

int WaveFunction::setTrialWF(int N, int M, double sigma_sqrd, double omega)
{
    m_N = N;
    m_M = M;
    m_sigma_sqrd = sigma_sqrd;
    m_omega_sqrd = omega*omega;
}

double WaveFunction::Psi_value_sqrd(VectorXd a, VectorXd b, VectorXd X, MatrixXd W)
{
    //Unnormalized wave function

    VectorXd v = b + (X.transpose() * W).transpose()/m_sigma_sqrd;
    VectorXd Xa = X - a;

    double prod = 1;
    for(int i=0; i<m_N; i++) {
        prod *= (1 + exp(v(i)));
    }

    return exp(-(double) (Xa.transpose() * Xa)/(m_sigma_sqrd)) * prod * prod;
}

double WaveFunction::Psi_value_sqrd_hastings(VectorXd Xa, VectorXd v)
{
    //Unnormalized wave function

    double prod = 1;
    for(int i=0; i<m_N; i++) {
        prod *= (1 + exp(v(i)));
    }

    return exp(-(double) (Xa.transpose() * Xa)/(m_sigma_sqrd)) * prod * prod;
}

double WaveFunction::EL_calc(VectorXd X, VectorXd Xa, VectorXd v, MatrixXd W, int D, int interaction, double &E_k, double &E_ext, double &E_int) {
    // Local energy calculations

    // Kinetic energy
    VectorXd e = VectorXd::Zero(m_N);
    for(int i=0; i<m_N; i++) {
        e(i) = 1/(1 + exp(-v(i)));
    }

    double E = 0;
    for(int i=0; i<m_N; i++) {
        E += (double) (Xa.transpose() * W.col(i)) * e(i);
        E += (double) ((W.col(i)).transpose() * W.col(i)) * e(i) * e(i);
        for(int j=0; j<m_N; j++) {
            E += (double) ((W.col(i)).transpose() * W.col(j)) * e(i) * e(j);
        }
    }

    E -= m_N * m_sigma_sqrd;
    E += Xa.transpose() * Xa;
    E = E/(2 * m_sigma_sqrd * m_sigma_sqrd);
    E_k += E;

    // Interaction energy
    double E_p = 0;
    if(interaction) E_p += rij(X, D);
    E += E_p;
    E_int += E_p;

    // Harmonic oscillator potential
    E_p = (double) (X.transpose() * X) * m_omega_sqrd/ 2;
    E += E_p;
    E_ext += E_p;

    return E;
}

void WaveFunction::Gradient_a(VectorXd Xa, VectorXd &da) {

    da = Xa/m_sigma_sqrd;
}

void WaveFunction::Gradient_b(VectorXd v, VectorXd &db) {

    for(int i=0; i<m_N; i++)
        db(i) = 1/(1 + exp(-v(i)));
}

void WaveFunction::Gradient_W(VectorXd X, VectorXd v, MatrixXd &dW) {

    for(int i=0; i<m_N; i++) {
        for(int j=0; j<m_N; j++) {
            dW(i,j) = X(i)/(1 + exp(-v(j)));
        }
    }
}
