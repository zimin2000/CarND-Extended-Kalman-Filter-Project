#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;

    F_ = F_in;

    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;

    I_ = MatrixXd::Identity(x_.size(), x_.size());
}

// Predict the state.
void KalmanFilter::Predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

// Update the state.
void KalmanFilter::Update(const VectorXd &z) {
    VectorXd z_pred = H_ * x_;

    VectorXd y = z - z_pred;

    MatrixXd Ht = H_.transpose();

    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    // New estimate
    x_ = x_ + (K * y);
    P_ = (I_ - K * H_) * P_;
}

// Update the state by using Extended Kalman Filter equations.
void KalmanFilter::UpdateEKF(const VectorXd &z) {
    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];

    float rho = sqrt(px * px + py * py);
    float phi = atan2(py, px);
    float drho = (px*vx + py*vy) / rho;

    VectorXd z_pred(3);
    z_pred << rho, phi, drho;

    VectorXd y = z - z_pred;

    // To keep rho within [-M_PI;M_PI] range.
    if (y[1] > M_PI) y[1] -= 2. * M_PI;
    else if (y[1] < -M_PI) y[1] += 2. * M_PI;

    MatrixXd Ht = H_.transpose();

    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    // New estimate
    x_ = x_ + (K * y);
    P_ = (I_ - K * H_) * P_;
}
