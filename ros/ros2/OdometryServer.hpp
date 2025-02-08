#pragma once

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS2
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_broadcaster.h"

// Eigen & Sophus
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <cmath>
#include <deque>

namespace kiss_icp_ros {

//
// Simple 2D EKF class (state: [x, y, theta])
//
class EKF2D
{
public:
  EKF2D()
  {
    state_.setZero(); // 초기 상태: [0, 0, 0]
    P_.setIdentity();
    P_ *= 1.0;

    // Process noise Q
    Q_.setZero();
    Q_(0,0) = 0.01 / 100.0;
    Q_(1,1) = 0.01 / 100.0;
    Q_(2,2) = 0.01 / 100.0;

    // Measurement noise R
    R_.setZero();
    R_(0,0) = 0.02;
    R_(1,1) = 0.02;
    R_(2,2) = 0.02;
  }

  Eigen::Vector3d getState() const { return state_; }

  // Predict step: integrate wheel odometry (v: linear, w: angular, dt: time interval)
  void predict(double v, double w, double dt)
  {
    double x = state_(0);
    double y = state_(1);
    double theta = state_(2);

    double new_x = x + v * std::cos(theta) * dt;
    double new_y = y + v * std::sin(theta) * dt;
    double new_theta = theta + w * dt;

    state_ << new_x, new_y, new_theta;

    // Jacobian of the motion model.
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0,2) = -v * std::sin(theta) * dt;
    F(1,2) =  v * std::cos(theta) * dt;

    P_ = F * P_ * F.transpose() + Q_;
  }

  // Update step: fuse a measurement (x, y, theta)
  void update(double x_meas, double y_meas, double theta_meas)
  {
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Vector3d z(x_meas, y_meas, theta_meas);
    Eigen::Vector3d z_pred = H * state_;

    Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
    Eigen::Matrix3d K = P_ * H.transpose() * S.inverse();

    state_ = state_ + K * (z - z_pred);
    P_ = (Eigen::Matrix3d::Identity() - K * H) * P_;
  }

private:
  Eigen::Vector3d state_; // [x, y, theta]
  Eigen::Matrix3d P_;
  Eigen::Matrix3d Q_;
  Eigen::Matrix3d R_;
};

class OdometryServer : public rclcpp::Node {
public:
    /// Constructor.
    OdometryServer();

private:
    /// LiDAR callback: perform ICP registration and EKF update.
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    /// Wheel odometry callback: perform EKF prediction.
    void wheelOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

private:
    // ROS node settings.
    size_t queue_size_{1};

    // TF broadcaster.
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Subscribers.
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;

    // Publishers.
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kpoints_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;

    // Path publisher.
    nav_msgs::msg::Path path_msg_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_publisher_;

    // KISS-ICP pipeline and configuration.
    kiss_icp::pipeline::KissICP odometry_;
    kiss_icp::pipeline::KISSConfig config_;

    // Global / map coordinate frames.
    std::string odom_frame_{"odom_LW"};
    std::string child_frame_{"base_link_LW"};

    // EKF filter for 2D state.
    EKF2D ekf2d_;

    // Last wheel odometry timestamp.
    rclcpp::Time last_wheel_stamp_;
};

}  // namespace kiss_icp_ros
