#include "ekf_centralized.h"
#include <cmath>
#include <iomanip>

EkfCentralized::EkfCentralized(int num_robots, const Eigen::VectorXd& initial_state) : 
    num_robots_(num_robots), state_(initial_state) {
    
    int state_size = num_robots_ * 3;
    P_ = Eigen::MatrixXd::Identity(state_size, state_size) * 0.1;
    
    // Process noise (tune these values)
    Q_ = (Eigen::Matrix3d() << 0.01, 0,    0,
                               0,    0.01, 0,
                               0,    0,    0.005).finished();
    
    // Measurement noise (tune these values)
    R_landmark_ = (Eigen::Matrix2d() << 0.1,  0,
                                        0,    0.05).finished();
    R_robot_ = (Eigen::Matrix2d() << 0.1, 0,
                                   0,   0.05).finished();
}

void EkfCentralized::setLandmarks(const std::map<int, Landmark>& landmarks) {
    landmark_map_ = landmarks;
}

void EkfCentralized::setState(const Eigen::VectorXd& new_state) {
    if(new_state.size() == state_.size()) {
        state_ = new_state;
    }
}

void EkfCentralized::predict(const OdometryData& odom, double dt) {
    int idx = (odom.robot_id - 1) * 3;
    Eigen::Vector3d state = state_.segment<3>(idx);
    
    // Motion model (unicycle)
    double theta = state[2];
    state[0] += odom.v * cos(theta) * dt;
    state[1] += odom.v * sin(theta) * dt;
    state[2] += odom.w * dt;
    state[2] = normalize_angle(state[2]);
    
    state_.segment<3>(idx) = state;
    
    // Jacobian of motion model
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0,2) = -odom.v * sin(theta) * dt;
    F(1,2) = odom.v * cos(theta) * dt;
    
    // Covariance update
    P_.block<3,3>(idx, idx) = F * P_.block<3,3>(idx, idx) * F.transpose() + Q_;
}

bool EkfCentralized::validateMeasurement(const MeasurementData& meas, double mahalanobis_threshold) {
    if(meas.subject_id <= 0) return false;

    int observer_idx = (meas.observer_id - 1) * 3;
    Eigen::Vector3d obs = state_.segment<3>(observer_idx);

    Eigen::Vector2d z_pred;
    Eigen::MatrixXd H;
    Eigen::Matrix2d R;

    if(meas.subject_id > num_robots_) {
        auto it = landmark_map_.find(meas.subject_id);
        if(it == landmark_map_.end()) return false;
        Eigen::Vector2d landmark = it->second.pos;
        double dx = landmark[0] - obs[0];
        double dy = landmark[1] - obs[1];
        double range = sqrt(dx*dx + dy*dy);
        double bearing = atan2(dy, dx) - obs[2];
        bearing = normalize_angle(bearing);
        z_pred << range, bearing;
        H = Eigen::MatrixXd::Zero(2, num_robots_*3);
        H.block<2,3>(0, observer_idx) << -dx/range, -dy/range, 0,
                                         dy/(range*range), -dx/(range*range), -1;
        R = R_landmark_;
    } else if(meas.subject_id != meas.observer_id) {
        int subject_idx = (meas.subject_id - 1) * 3;
        Eigen::Vector3d sub = state_.segment<3>(subject_idx);
        double dx = sub[0] - obs[0];
        double dy = sub[1] - obs[1];
        double range = sqrt(dx*dx + dy*dy);
        double bearing = atan2(dy, dx) - obs[2];
        bearing = normalize_angle(bearing);
        z_pred << range, bearing;
        H = Eigen::MatrixXd::Zero(2, num_robots_*3);
        H.block<2,3>(0, observer_idx) << -dx/range, -dy/range, 0,
                                         dy/(range*range), -dx/(range*range), -1;
        H.block<2,3>(0, subject_idx) << dx/range, dy/range, 0,
                                       -dy/(range*range), dx/(range*range), 0;
        R = R_robot_;
    } else {
        return false;
    }

    Eigen::Vector2d z(meas.range, meas.bearing);
    Eigen::Vector2d dz = z - z_pred;
    dz[1] = normalize_angle(dz[1]);

    Eigen::Matrix2d S = H * P_ * H.transpose() + R;
    double mahalanobis = sqrt(dz.transpose() * S.inverse() * dz);
    return mahalanobis < mahalanobis_threshold;
}

void EkfCentralized::correct(const MeasurementData& meas) {
    if(!validateMeasurement(meas)) return;

    int state_size = num_robots_ * 3;
    int obs_idx = (meas.observer_id - 1) * 3;

    Eigen::VectorXd z_pred(2);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, state_size);
    Eigen::Matrix2d R;

    if(meas.subject_id > num_robots_) {
        Eigen::Vector3d obs = state_.segment<3>(obs_idx);
        Eigen::Vector2d landmark = landmark_map_[meas.subject_id].pos;
        double dx = landmark[0] - obs[0];
        double dy = landmark[1] - obs[1];
        double range = std::sqrt(dx*dx + dy*dy);
        double bearing = std::atan2(dy, dx) - obs[2];
        bearing = normalize_angle(bearing);
        z_pred << range, bearing;
        H.block<2,3>(0, obs_idx) << -dx/range, -dy/range, 0,
                                   dy/(range*range), -dx/(range*range), -1;
        R = R_landmark_;
    } else {
        int sub_idx = (meas.subject_id - 1) * 3;
        Eigen::Vector3d obs = state_.segment<3>(obs_idx);
        Eigen::Vector3d sub = state_.segment<3>(sub_idx);
        double dx = sub[0] - obs[0];
        double dy = sub[1] - obs[1];
        double range = std::sqrt(dx*dx + dy*dy);
        double bearing = std::atan2(dy, dx) - obs[2];
        bearing = normalize_angle(bearing);
        z_pred << range, bearing;
        H.block<2,3>(0, obs_idx) << -dx/range, -dy/range, 0,
                                   dy/(range*range), -dx/(range*range), -1;
        H.block<2,3>(0, sub_idx) << dx/range, dy/range, 0,
                                   -dy/(range*range), dx/(range*range), 0;
        R = R_robot_;
    }

    Eigen::Vector2d z(meas.range, meas.bearing);
    Eigen::Vector2d dz = z - z_pred;
    dz[1] = normalize_angle(dz[1]);

    Eigen::Matrix2d S = H * P_ * H.transpose() + R;
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();

    state_ += K * dz;
    for(int r = 0; r < num_robots_; ++r) {
        state_[3*r + 2] = normalize_angle(state_[3*r + 2]);
    }

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(state_size, state_size);
    P_ = (I - K * H) * P_ * (I - K * H).transpose() + K * R * K.transpose();

    last_correction_time_ = meas.timestamp;
}

void EkfCentralized::printState(rclcpp::Logger logger, double timestamp) const {
    std::stringstream ss;
    ss << "\nTime " << std::fixed << std::setprecision(3) << timestamp;
    ss << "\nCurrent state:";
    for(int i = 0; i < num_robots_; ++i) {
        ss << "\nRobot " << i+1 << ": "
           << std::fixed << std::setprecision(3)
           << "x=" << state_(3*i) 
           << " y=" << state_(3*i+1)
           << " θ=" << state_(3*i+2);
    }
    RCLCPP_INFO(logger, "%s", ss.str().c_str());
}

void EkfCentralized::printUncertainty(rclcpp::Logger logger) const {
    std::stringstream ss;
    ss << "\nUncertainty (diagonal elements):";
    for(int i = 0; i < num_robots_; ++i) {
        ss << "\nRobot " << i+1 << ": "
           << std::fixed << std::setprecision(5)
           << "σ²_x=" << P_(3*i, 3*i)
           << " σ²_y=" << P_(3*i+1, 3*i+1)
           << " σ²_θ=" << P_(3*i+2, 3*i+2);
    }
    RCLCPP_DEBUG(logger, "%s", ss.str().c_str());
}

double EkfCentralized::normalize_angle(double angle) {
    while(angle > M_PI) angle -= 2*M_PI;
    while(angle < -M_PI) angle += 2*M_PI;
    return angle;
}