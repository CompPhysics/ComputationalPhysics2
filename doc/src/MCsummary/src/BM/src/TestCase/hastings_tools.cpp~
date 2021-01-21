#include <iostream>
#include "wavefunction.h"
#include "hastings_tools.h"
#include <random>
#include "eigen3/Eigen/Dense"
#include <cmath>

using namespace Eigen;
using namespace std;

double QForce(const VectorXd &Xa, const VectorXd &v, const MatrixXd &W, double sigma_sqrd, int i) {

    int N = v.size();
    double QF = -Xa(i);
    for(int j=0; j<N; j++) {
        QF += W(i,j)/(1 + exp(-v(j)));
    }
    return QF*(2/(sigma_sqrd));
}

double GreenFuncSum(const VectorXd &X, const VectorXd &X_new, const VectorXd &X_newa, const VectorXd &Xa, const VectorXd &v,\
                    const MatrixXd &W, double sigma_sqrd, double timestep, int D, double Diff) {
    double GreenSum  = 0;
    double QForceOld = 0;
    double QForceNew = 0;

    int P = X.size()/D;

    for(int i=0; i<P; i++) {
        double GreenFunc = 0;
        for(int j=0; j<D; j++) {
            QForceOld = QForce(Xa, v, W, sigma_sqrd, D*i+j);
            QForceNew = QForce(X_newa, v, W, sigma_sqrd, D*i+j);
            GreenFunc += 0.5*(QForceOld + QForceNew) * (0.5*Diff*timestep*(QForceOld - QForceNew)-X_new(D*i+j)+X(D*i+j));
        }
        GreenSum += exp(GreenFunc);
    }

    return GreenSum;
}
