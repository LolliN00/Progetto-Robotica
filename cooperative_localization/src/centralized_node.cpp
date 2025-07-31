#include "rclcpp/rclcpp.hpp"
#include "data_loader.h"
#include "ekf_centralized.h"
#include <memory>
#include <iomanip>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("centralized_ekf_node");
    
    // 1. Load data - MODIFY THIS PATH!
    std::string base_path = "/home/leonardo/asrl3.utias.utoronto.ca/MRCLAM1/MRCLAM_Dataset1/";
    DataLoader loader;
    
    // Load landmarks
    if(!loader.loadLandmarkGroundtruth(base_path + "Landmark_Groundtruth.dat")) {
        RCLCPP_ERROR(node->get_logger(), "Failed to load landmarks!");
        return 1;
    }
    
    // Load data for 2 robots
    for(int i = 1; i <= 2; ++i) {
        if(!loader.loadOdometry(base_path + "Robot" + std::to_string(i) + "_Odometry.dat", i)) {
            RCLCPP_ERROR(node->get_logger(), "Failed to load odometry for Robot %d!", i);
        }
        if(!loader.loadMeasurements(base_path + "Robot" + std::to_string(i) + "_Measurement.dat", i)) {
            RCLCPP_ERROR(node->get_logger(), "Failed to load measurements for Robot %d!", i);
        }
    }
    
    auto events = loader.getEventQueue();
    RCLCPP_INFO(node->get_logger(), "Loaded %zu events", events.size());
    
    // 2. Initialize EKF with groundtruth
    Eigen::VectorXd initial_state(6);
    // Robot 1: [3.573, -3.332, 2.340]
    // Robot 2: [0.623, -1.432, 1.346]
    initial_state << 3.573, -3.332, 2.340, 
                    0.623, -1.432, 1.346;
    
    EkfCentralized ekf(2, initial_state);
    ekf.setLandmarks(loader.getLandmarks());
    
    // 3. Main processing loop
    double last_time = -1;
    size_t processed = 0;
    for(const auto& event : events) {
        double t = std::visit([](auto&& e){ return e.timestamp; }, event);
        if(last_time < 0) {
            last_time = t;
            continue;
        }
        
        double dt = t - last_time;
        std::visit([&](auto&& e) {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, OdometryData>) {
                ekf.predict(e, dt);
            } else if constexpr (std::is_same_v<T, MeasurementData>) {
                if(e.subject_id > 0) { // Only landmarks
                    ekf.correct(e);
                }
            }
        }, event);
        
        last_time = t;
        processed++;
        
        // Print every 100 events
        if(processed % 100 == 0) {
            ekf.printState(node->get_logger());
            ekf.printUncertainty(node->get_logger());
        }
    }
    
    // Final state
    RCLCPP_INFO(node->get_logger(), "Final state:");
    ekf.printState(node->get_logger());
    ekf.printUncertainty(node->get_logger());
    
    RCLCPP_INFO(node->get_logger(), "Processing completed!");
    rclcpp::shutdown();
    return 0;
}