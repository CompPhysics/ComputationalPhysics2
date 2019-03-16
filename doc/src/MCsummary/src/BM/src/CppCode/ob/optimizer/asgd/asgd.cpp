#include "asgd.h"

Asgd::Asgd(double a, double A, double asgdOmega, double fmax, double fmin,
           double t0, double t1, int nPar) : Optimizer() {
    m_a = a;
    m_A = A;
    m_omega = asgdOmega;
    m_fmax = fmax;
    m_fmin = fmin;
    m_t = t1;
    m_tprev = t0;

    // Setting to 0 so that the first update of t will be t=tprev+f=tprev.
    m_gradPrev = Eigen::VectorXd::Zero(nPar);
}

Asgd::Asgd(double a, double A, double nPar) : Optimizer() {
    m_a = a;
    m_A = A;
    m_omega = 1.0;
    m_fmax = 2.0;
    m_fmin = -0.5;
    m_t = m_A;
    m_tprev = m_A;

    // Setting to 0 so that the first update of t will be t=tprev+f=tprev.
    m_gradPrev = Eigen::VectorXd::Zero(nPar);
}


void Asgd::optimizeWeights(NeuralQuantumState &nqs, Eigen::VectorXd grad, int cycles) {
    /* Calculate the learning rate: gamma = a/(t_i + A)
     * Calculate the new t=max(0, tprev + f), with f responsible for altering the learning rate
     * according to changes in the gradient:
     *
     * If we passed a minimum then the negative product between
     * the current and previous gradient, gradProduct, will be positive. Then f in [0, fmax]
     * and t=tprev + f. This causes an increase in t, causing the learning rate to decrease.
     *
     * If the gradient in two conecutive steps point in the same direction, gradProduct is
     * negative, and f in [fmin, 0]. This reduces t, causing the learning rate to increase.
     * If f is so small that f<-tprev, then t=0 and the learning rate is gamma=a/A, its maximum.
     *
     * To clarify: t_i is here based on the product of gradients from iteration i and i-1.
     * Note: this implementation only really uses the t0 user setting. t1 is here updated according
     * to the algorithm, not the user's choice. m_tprev does not need to be a variable in this
     * implementation. Have still kept both t0 and t1 as input parameters for now as it's mentioned in
     * the algorithm description, in case of future changes. */

    double gradProduct = -grad.dot(m_gradPrev);
    double f = m_fmin + (m_fmax - m_fmin)/(1 - (m_fmax/m_fmin)*exp(-gradProduct/m_omega));
    double tnext = m_tprev + f;
    // Update m_t
    m_t = 0.0;
    if (0.0 < tnext) m_t=tnext;
    // Compute the learning rate
    double gamma = m_a/(m_t+m_A);


    // Compute new parameters
    for (int i=0; i<nqs.m_nx; i++) {
        nqs.m_a(i) = nqs.m_a(i) - gamma*grad(i);
    }
    for (int j=0; j<nqs.m_nh; j++) {
        nqs.m_b(j) = nqs.m_b(j) - gamma*grad(nqs.m_nx + j);
    }
    int k = nqs.m_nx + nqs.m_nh;
    for (int i=0; i<nqs.m_nx; i++) {
        for (int j=0; j<nqs.m_nh; j++) {
            nqs.m_w(i,j) = nqs.m_w(i,j) - gamma*grad(k);
            k++;
        }
    }

    m_gradPrev = grad;
    m_tprev = m_t;

}

