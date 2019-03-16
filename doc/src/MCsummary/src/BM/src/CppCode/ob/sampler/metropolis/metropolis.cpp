#include "metropolis.h"
#include "math.h"

Metropolis::Metropolis(int nSamples, int nCycles, double step, Hamiltonian &hamiltonian,
                       NeuralQuantumState &nqs, Optimizer &optimizer,
                       std::string filename, std::string blockFilename, int seed) :
    Sampler(nSamples, nCycles, hamiltonian, nqs, optimizer, filename, blockFilename, seed) {
    m_psi = m_nqs.computePsi(); // Set the Psi variable to correspond to the initial positions
    m_accepted = 0.0;
    m_step = step;
    m_distributionStep = std::uniform_real_distribution<double>(-0.5, 0.5);
    m_distributionTest = std::uniform_real_distribution<double>(0.0, 1.0);
    m_distributionImportance = std::normal_distribution<double>(0.0, 1.0);
}

// Update all coordinates of all particles pr sampling
/*
void Metropolis::samplePositions(int &accepted) {
    // Suggestion of new position xTrial
    Eigen::VectorXd xTrial;
    xTrial.resize(m_nqs.m_nx);
    for (int i=0; i<m_nqs.m_nx; i++) {
        xTrial(i) = m_nqs.m_x(i) + m_distributionStep(m_randomEngine)*m_step;
    }

    double psiTrial = m_nqs.computePsi(xTrial);

    double probCurrent = m_psi*m_psi;
    double probTrial = psiTrial*psiTrial;\
    double probRatio = probTrial/probCurrent;

    if ((1.0 < probRatio) || (m_distributionTest(m_randomEngine) < probRatio)) {
        m_nqs.m_x = xTrial;
        m_psi = psiTrial;
        //m_accepted++;
        accepted++;
    }
}
*/

// Update one coordinate at a time (one coordinate pr one sampling)
void Metropolis::samplePositions(bool &accepted) {
    // Suggestion of new position xTrial
    Eigen::VectorXd xTrial = m_nqs.m_x;

    std::uniform_int_distribution<> mrand(0, m_nqs.m_nx-1);
    int updateCoordinate = mrand(m_randomEngine);

    /*  // brute force
    xTrial(updateCoordinate) += m_distributionStep(m_randomEngine)*m_step;

    double psiTrial = m_nqs.computePsi(xTrial);

    double probCurrent = m_psi*m_psi;
    double probTrial = psiTrial*psiTrial;
    double probRatio = probTrial/probCurrent;

    if ((1.0 < probRatio) || (m_distributionTest(m_randomEngine) < probRatio)) {
        m_nqs.m_x = xTrial;
        m_psi = psiTrial;
        //m_accepted++;
        accepted++;
    }
    */
    double D = 0.5;
    double xi = m_distributionImportance(m_randomEngine);
    double Fcurrent;
    double Ftrial;

    // Compute quantum force
    Eigen::VectorXd Q = m_nqs.m_b + (1.0/m_nqs.m_sig2)*(m_nqs.m_x.transpose()*m_nqs.m_w).transpose();
    double sum1 = 0;
    for (int j=0; j<m_nqs.m_nh; j++) {
        sum1 += m_nqs.m_w(updateCoordinate,j)/(1.0+exp(-Q(j)));
    }
    Fcurrent = 2*(-(m_nqs.m_x(updateCoordinate) - m_nqs.m_a(updateCoordinate))/m_nqs.m_sig2 + sum1/m_nqs.m_sig2);

    // Update coordinate
    double xCurrent = m_nqs.m_x(updateCoordinate);
    xTrial(updateCoordinate) = xCurrent + D*Fcurrent*m_step + xi*sqrt(m_step);


    Q = m_nqs.m_b + (1.0/m_nqs.m_sig2)*(xTrial.transpose()*m_nqs.m_w).transpose();
    sum1 = 0;
    for (int j=0; j<m_nqs.m_nh; j++) {
        sum1 += m_nqs.m_w(updateCoordinate,j)/(1.0+exp(-Q(j)));
    }
    Ftrial = 2*(-(xTrial(updateCoordinate) - m_nqs.m_a(updateCoordinate))/m_nqs.m_sig2 + sum1/m_nqs.m_sig2);

    //Greens ratio
    double part1 = xCurrent - xTrial(updateCoordinate) - m_step*Ftrial;
    double part2 = xTrial(updateCoordinate) - xCurrent - m_step*Fcurrent;
    double Gratio = exp(-(part1*part1 - part2*part2)/(4*D*m_step));

    // Psi
    double psiTrial = m_nqs.computePsi(xTrial);
    double probCurrent = m_psi*m_psi;

    double probRatio = Gratio*psiTrial*psiTrial/(m_psi*m_psi);

    if ((1.0 < probRatio) || (m_distributionTest(m_randomEngine) < probRatio)) {
        m_nqs.m_x = xTrial;
        m_psi = psiTrial;
        //m_accepted++;
        accepted=true;
    } else {
        accepted=false;
    }

}

