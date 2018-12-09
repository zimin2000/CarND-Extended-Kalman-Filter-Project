#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    assert(ground_truth.size() == estimations.size());
    assert(estimations.size() > 0);

    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // Accumulate squared residuals
    for (size_t i = 0; i < estimations.size(); ++i) {

        VectorXd residual = estimations[i] - ground_truth[i];

        // Coefficient-wise multiplication
        residual = residual.array() * residual.array();

        rmse += residual;
    }

    rmse = rmse / estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x) {

    float px = x[0];
    float py = x[1];
    float vx = x[2];
    float vy = x[3];

    assert (px != 0. || py != 0.);

    // Compute the Jacobian matrix
    float r2 = px*px + py*py;
    float r = sqrt(r2);
    float r3 = r2 * r;

    float v = vx*py - vy*px;

    MatrixXd Hj(3, 4);
    Hj <<   px / r,     py / r,     0.,     0.,
            -py / r2,   px / r2,    0.,     0.,
            py*v / r3,  -px*v / r3, px / r, py / r;

    return Hj;
}

