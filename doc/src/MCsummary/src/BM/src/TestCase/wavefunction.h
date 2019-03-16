#pragma once
#include <vector>
#include "../Eigen/Dense"

using namespace Eigen;

class WaveFunction
{
private:
    int m_M;
    int m_N;
    double m_sigma_sqrd;
    double m_omega_sqrd;
public:
    WaveFunction() {}
    int setTrialWF              (int N, int M, double sigma_sqrd, double omega);
    double Psi_value_sqrd(VectorXd a, VectorXd b, VectorXd X, MatrixXd W);
    double Psi_value_sqrd_hastings(VectorXd Xa, VectorXd v);
    double EL_calc(VectorXd X, VectorXd Xa, VectorXd v, MatrixXd W, int D, int interaction, double &E_k, double &E_ext, double &E_int);
    void Gradient_a(VectorXd a, VectorXd &da);
    void Gradient_b(VectorXd b, VectorXd &db);
    void Gradient_W(VectorXd X, VectorXd v, MatrixXd &dW);
};
