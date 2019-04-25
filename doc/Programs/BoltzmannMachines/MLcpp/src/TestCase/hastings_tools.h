#pragma once
#include "eigen3/Eigen/Dense"

double QForce(const Eigen::VectorXd &Xa, const Eigen::VectorXd &v, const Eigen::MatrixXd &W, double sigma_sqrd, int i);
double GreenFuncSum(const Eigen::VectorXd &X, const Eigen::VectorXd &X_new, const Eigen::VectorXd &X_newa, const Eigen::VectorXd &Xa, \
                    const Eigen::VectorXd &v, const Eigen::MatrixXd &W, double sigma, double timestep, int D, double Diff);
