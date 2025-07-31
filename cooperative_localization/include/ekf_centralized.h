#ifndef EKF_CENTRALIZED_H
#define EKF_CENTRALIZED_H

#include <Eigen/Dense>
#include <map>
#include <vector>
#include "data_loader.h"
#include <rclcpp/rclcpp.hpp>

class EkfCentralized {
public:
    EkfCentralized(int num_robots, const Eigen::VectorXd& initial_state);
    void setLandmarks(const std::map<int, Landmark>& landmarks);
    void predict(const OdometryData& odom, double dt);
    void correct(const MeasurementData& meas);
    
    const Eigen::VectorXd& getState() const { return state_; }
    const Eigen::MatrixXd& getCovariance() const { return P_; }
    void printState(rclcpp::Logger logger, double timestamp) const;
    void printUncertainty(rclcpp::Logger logger) const;
    void setState(const Eigen::VectorXd& new_state);

private:
    double normalize_angle(double angle);
    bool validateMeasurement(const MeasurementData& meas, double mahalanobis_threshold = 3.0);
    
    Eigen::VectorXd state_;
    Eigen::MatrixXd P_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_landmark_;
    Eigen::MatrixXd R_robot_;
    
    int num_robots_;
    std::map<int, Landmark> landmark_map_;
    double last_correction_time_ = -1;
};

#endif