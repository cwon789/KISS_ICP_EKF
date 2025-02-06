// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS2
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "tf2_ros/transform_broadcaster.h"

// For the EKF2D filter
#include <Eigen/Dense>
#include <cmath>

//
// Simple 2D EKF class (state: [x, y, theta])
//
class EKF2D
{
public:
  EKF2D()
  {
    state_.setZero(); // [x=0, y=0, theta=0]
    P_.setIdentity();
    P_ *= 1.0;

    // Process noise Q
    Q_.setZero();
    Q_(0,0) = 0.01;
    Q_(1,1) = 0.01;
    Q_(2,2) = 0.001;

    // Measurement noise R
    R_.setZero();
    R_(0,0) = 0.05;
    R_(1,1) = 0.05;
    R_(2,2) = 0.01;
  }

  Eigen::Vector3d getState() const { return state_; }

  // Predict step: integrates wheel odometry (v: linear, w: angular, dt: time interval)
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

  // Update step: fuses a measurement (x, y, theta)
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

namespace kiss_icp_ros {

class OdometryServer : public rclcpp::Node {
public:
    /// OdometryServer constructor
    OdometryServer();

private:
    /// Register new frame (LiDAR callback)
    void RegisterFrame(const sensor_msgs::msg::PointCloud2::SharedPtr msg_ptr);
    /// Wheel odometry callback
    void WheelOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

private:
    /// ROS node settings
    size_t queue_size_{1};

    /// Tools for broadcasting TFs.
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    /// Data subscribers.
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;

    /// Data publishers.
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr frame_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr kpoints_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;

    /// Path publisher.
    nav_msgs::msg::Path path_msg_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr traj_publisher_;

    /// KISS-ICP pipeline and configuration.
    kiss_icp::pipeline::KissICP odometry_;
    kiss_icp::pipeline::KISSConfig config_;

    /// Global / map coordinate frames.
    std::string odom_frame_{"odom"};
    std::string child_frame_{"base_link"};

    // =======================================================================
    // New members for the 2D EKF filter.
    // =======================================================================
    EKF2D ekf2d_;
    rclcpp::Time last_wheel_stamp_;
};

}  // namespace kiss_icp_ros
