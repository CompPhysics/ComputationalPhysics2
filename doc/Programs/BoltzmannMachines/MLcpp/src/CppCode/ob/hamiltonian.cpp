#include "hamiltonian.h"


Hamiltonian::Hamiltonian(double omega, bool includeInteraction) {
    m_omega = omega;
    m_includeInteraction = includeInteraction;
}


double Hamiltonian::computeLocalEnergy(NeuralQuantumState &nqs) {
    Eigen::VectorXd Q = nqs.m_b + (1.0/nqs.m_sig2)*(nqs.m_x.transpose()*nqs.m_w).transpose();
    double ElocTemp = 0;
    // Loop over the visibles (n_particles*n_coordinates) for the Laplacian
    for (int r=0; r<nqs.m_nx; r++) {
        double sum1 = 0;
        double sum2 = 0;
        for (int j=0; j<nqs.m_nh; j++) {
            sum1 += nqs.m_w(r,j)/(1.0+exp(-Q(j)));
            sum2 += nqs.m_w(r,j)*nqs.m_w(r,j)*exp(Q(j))/((exp(Q(j))+1.0)*(exp(Q(j))+1.0));
        }
        double der1lnPsi = -(nqs.m_x(r) - nqs.m_a(r))/nqs.m_sig2 + sum1/nqs.m_sig2;
        double der2lnPsi = -1.0/nqs.m_sig2 + sum2/(nqs.m_sig2*nqs.m_sig2);
        // The last term is the Harmonic Oscillator potential
        ElocTemp += -der1lnPsi*der1lnPsi - der2lnPsi + m_omega*m_omega*nqs.m_x(r)*nqs.m_x(r);


    }
    ElocTemp = 0.5*ElocTemp;

    // With interaction:
    if (m_includeInteraction) {
        ElocTemp += interaction(nqs.m_x, nqs.m_nx, nqs.m_dim);
    }

    return ElocTemp;
}

Eigen::VectorXd Hamiltonian::computeLocalEnergyGradientComponent(NeuralQuantumState &nqs) {

    // Compute the 1/psi * dPsi/dalpha_i, that is Psi derived wrt each RBM parameter.
    Eigen::VectorXd Q = nqs.m_b + (1.0/nqs.m_sig2)*(nqs.m_x.transpose()*nqs.m_w).transpose();
    Eigen::VectorXd derPsiTemp;
    derPsiTemp.resize(nqs.m_nx + nqs.m_nh + nqs.m_nx*nqs.m_nh);

    for (int k=0; k<nqs.m_nx; k++) {
        derPsiTemp(k) = (nqs.m_x(k) - nqs.m_a(k))/nqs.m_sig2;
    }
    for (int k=nqs.m_nx; k<(nqs.m_nx+nqs.m_nh); k++) {
        derPsiTemp(k) = 1.0/(1.0+exp(-Q(k-nqs.m_nx)));
    }
    int k=nqs.m_nx + nqs.m_nh;
    for (int i=0; i<nqs.m_nx; i++) {
        for (int j=0; j<nqs.m_nh; j++) {
            derPsiTemp(k) = nqs.m_x(i)/(nqs.m_sig2*(1.0+exp(-Q(j))));
            k++;
        }
    }
    return derPsiTemp;
}

double Hamiltonian::interaction(Eigen::VectorXd x, int nx, int dim) {
    double interactionTerm = 0;
    double rDistance;
    // Loop over each particle
    for (int r=0; r<nx-dim; r+=dim) {
        // Loop over each particle s that particle r hasn't been paired with
        for (int s=(r+dim); s<nx; s+=dim) {
            rDistance = 0;
            // Loop over each dimension
            for (int i=0; i<dim; i++) {
                rDistance += (x(r+i) - x(s+i))*(x(r+i) - x(s+i));
            }
            interactionTerm += 1.0/sqrt(rDistance);
        }
    }
    return interactionTerm;
}
