#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    // Initializing matrices.

    // Laser:
    //   Measurement covariance matrix.
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << 0.0225, 0,
                0,      0.0225;

    //   Observation matrix.
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    // Radar:
    //   Measurement covariance matrix.
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << 0.09, 0,      0,
                0,    0.0009, 0,
                0,    0,      0.09;

    //   Observation Jacobian (to be filled on update).
    Hj_ = MatrixXd(3, 4);

    // The EKF initialization postponed till the first update.
}

FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        previous_timestamp_ = measurement_pack.timestamp_;

        // Initialize the EKF with setting up the state for first measurement,
        // and filling covariance and proccess noise matrices.
        VectorXd x0(4);

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert radar from polar to cartesian coordinates and initialize state.
            float rho = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float px = rho*cos(phi);
            float py = rho*sin(phi);

            x0 << px, py, 0, 0;

        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            // Take initial state from laser measurements.
            float px = measurement_pack.raw_measurements_[0];
            float py = measurement_pack.raw_measurements_[1];

            x0 << px, py, 0, 0;
        }

        MatrixXd Q0 = MatrixXd(4, 4);
        Q0 <<   0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0;

        MatrixXd F0(4,4);
        F0 <<   1,  0,  0,  0,
                0,  1,  0,  0,
                0,  0,  1,  0,
                0,  0,  0,  1;

        MatrixXd P0(4,4);
        P0 <<   1,  0,  0,    0,
                0,  1,  0,    0,
                0,  0,  1000, 0,
                0,  0,  0,    1000;

        ekf_.Init(x0, P0, F0, H_laser_, R_laser_, Q0);

        // Done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
    *  Prediction
    ****************************************************************************/

    // Update the state transition matrix F according to the new elapsed time.
    //    - Time is measured in seconds.
    // Update the process noise covariance matrix.
    // Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    //
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;   //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

    float dt_2 = dt * dt;
    float dt_3 = dt_2 * dt;
    float dt_4 = dt_2 * dt_2;

    // Modify the F matrix so that the time is integrated
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    float noise_ax = 9;
    float noise_ay = 9;

    // Set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ <<  dt_4 / 4 * noise_ax,    0,                      dt_3 / 2 * noise_ax,    0,
                0,                      dt_4 / 4 * noise_ay,    0,                      dt_3 / 2 * noise_ay,
                dt_3 / 2 * noise_ax,    0,                      dt_2 * noise_ax,        0,
                0,                      dt_3 / 2 * noise_ay,    0,                      dt_2 * noise_ay;

    ekf_.Predict();

    /*****************************************************************************
    *  Update
    ****************************************************************************/

    // Use the sensor type to perform the update step.
    // Update the state and covariance matrices.
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates

        // Check whether px==0 & py==0, which will give error in calculating the Jacobian
        // skip this measurement.
        if (ekf_.x_[0] == 0 && ekf_.x_[1] == 0) {
            return;
        }

        ekf_.R_ = R_radar_;
        Hj_ = Tools::CalculateJacobian(ekf_.x_);
        ekf_.H_ = Hj_;

        ekf_.UpdateEKF(measurement_pack.raw_measurements_);

    } else {
        // Laser updates
        ekf_.R_ = R_laser_;
        ekf_.H_ = H_laser_;

        ekf_.Update(measurement_pack.raw_measurements_);
    }
}
