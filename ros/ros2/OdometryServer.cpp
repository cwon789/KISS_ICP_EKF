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
// (License text omitted for brevity)

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <cmath>
#include <vector>
#include <memory>

// Include EKF and utility functions.
#include "OdometryServer.hpp"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

// ROS2 headers
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2_ros/transform_broadcaster.h"

//
// -------------------------------------------------------------------------
// New 2D EKF class (from your reference code)
// -------------------------------------------------------------------------
// This class implements a simple EKF for 2D state [x, y, theta].
// Only the filter part has been changed; all other parts remain unchanged.
//
namespace {

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

  // Predict step using wheel odometry measurements: linear velocity v and angular velocity w.
  void predict(double v, double w, double dt)
  {
    double x = state_(0);
    double y = state_(1);
    double theta = state_(2);

    double new_x = x + v * std::cos(theta) * dt;
    double new_y = y + v * std::sin(theta) * dt;
    double new_theta = theta + w * dt;

    state_ << new_x, new_y, new_theta;

    // Jacobian F.
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0,2) = -v * std::sin(theta) * dt;
    F(1,2) =  v * std::cos(theta) * dt;

    P_ = F * P_ * F.transpose() + Q_;
  }

  // Update step using LiDAR measurement (x, y, theta).
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

}  // anonymous namespace

//
// -------------------------------------------------------------------------
// ROS2 Node: OdometryServer (modified to use 2D EKF filter)
// -------------------------------------------------------------------------
namespace kiss_icp_ros {

OdometryServer::OdometryServer() : rclcpp::Node("odometry_node"), odometry_(config_)
{
    // -------------------------------------------------------------------------
    // Parameter Initialization
    // -------------------------------------------------------------------------
    child_frame_ = declare_parameter<std::string>("child_frame", child_frame_);
    odom_frame_ = declare_parameter<std::string>("odom_frame", odom_frame_);
    config_.max_range = declare_parameter<double>("max_range", config_.max_range);
    config_.min_range = declare_parameter<double>("min_range", config_.min_range);
    config_.deskew = declare_parameter<bool>("deskew", config_.deskew);
    config_.voxel_size = declare_parameter<double>("voxel_size", config_.max_range / 100.0);
    config_.max_points_per_voxel = declare_parameter<int>("max_points_per_voxel", config_.max_points_per_voxel);
    config_.initial_threshold = declare_parameter<double>("initial_threshold", config_.initial_threshold);
    config_.min_motion_th = declare_parameter<double>("min_motion_th", config_.min_motion_th);
    if (config_.max_range < config_.min_range) {
        RCLCPP_WARN(get_logger(), "[WARNING] max_range is smaller than min_range, setting min_range to 0.0");
        config_.min_range = 0.0;
    }

    // Initialize the KISS-ICP pipeline.
    odometry_ = kiss_icp::pipeline::KissICP(config_);

    // -------------------------------------------------------------------------
    // Subscribers
    // -------------------------------------------------------------------------
    pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud_topic", rclcpp::SensorDataQoS(),
        std::bind(&OdometryServer::RegisterFrame, this, std::placeholders::_1));

    wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "wheel_odometry_topic", rclcpp::SensorDataQoS(),
        std::bind(&OdometryServer::WheelOdometryCallback, this, std::placeholders::_1));

    // -------------------------------------------------------------------------
    // Publishers
    // -------------------------------------------------------------------------
    rclcpp::QoS qos(rclcpp::KeepLast{queue_size_});
    odom_publisher_ = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
    frame_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("frame", qos);
    kpoints_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("keypoints", qos);
    map_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("local_map", qos);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    path_msg_.header.frame_id = odom_frame_;
    traj_publisher_ = create_publisher<nav_msgs::msg::Path>("trajectory", qos);

    // -------------------------------------------------------------------------
    // Broadcast static transform if needed.
    // -------------------------------------------------------------------------
    if (child_frame_ != "base_link") {
        static auto br = std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
        geometry_msgs::msg::TransformStamped alias_transform_msg;
        alias_transform_msg.header.stamp = this->get_clock()->now();
        alias_transform_msg.transform.translation.x = 0.0;
        alias_transform_msg.transform.translation.y = 0.0;
        alias_transform_msg.transform.translation.z = 0.0;
        alias_transform_msg.transform.rotation.x = 0.0;
        alias_transform_msg.transform.rotation.y = 0.0;
        alias_transform_msg.transform.rotation.z = 0.0;
        alias_transform_msg.transform.rotation.w = 1.0;
        alias_transform_msg.header.frame_id = child_frame_;
        alias_transform_msg.child_frame_id = "base_link";
        br->sendTransform(alias_transform_msg);
    }

    RCLCPP_INFO(this->get_logger(), "KISS-ICP ROS2 odometry node with EKF2D and time sync initialized");
}

//
// -----------------------------------------------------------------------------
// Wheel Odometry Callback (Prediction Step)
// -----------------------------------------------------------------------------
void OdometryServer::WheelOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Extract linear and angular velocities from wheel odometry.
    double v = msg->twist.twist.linear.x;    // [m/s]
    double w = msg->twist.twist.angular.z;     // [rad/s]
    rclcpp::Time stamp = msg->header.stamp;

    // Compute time difference.
    double dt = 0.0;
    if (last_wheel_stamp_.nanoseconds() != 0) {
        dt = (stamp - last_wheel_stamp_).seconds();
        if (dt < 0.0) {
            RCLCPP_WARN(this->get_logger(), "Negative dt in wheel odometry, setting dt=0");
            dt = 0.0;
        }
    }
    last_wheel_stamp_ = stamp;

    // EKF predict step.
    ekf2d_.predict(v, w, dt);
}

//
// -----------------------------------------------------------------------------
// LiDAR (ICP) Callback (Measurement Update)
// -----------------------------------------------------------------------------
void OdometryServer::RegisterFrame(const sensor_msgs::msg::PointCloud2::SharedPtr msg_ptr)
{
    const auto &msg = *msg_ptr;
    const auto points = kiss_icp_ros::utils::PointCloud2ToEigen(msg);
    const auto timestamps = [&]() -> std::vector<double> {
        if (!config_.deskew) return {};
        return kiss_icp_ros::utils::GetTimestamps(msg);
    }();

    // Create an initial guess for ICP registration from the current EKF state.
    Eigen::Vector3d ekf_state = ekf2d_.getState(); // [x, y, theta]
    Eigen::Isometry3d init_guess = Eigen::Isometry3d::Identity();
    init_guess.pretranslate(Eigen::Vector3d(ekf_state(0), ekf_state(1), 0.0));
    init_guess.rotate(Eigen::AngleAxisd(ekf_state(2), Eigen::Vector3d::UnitZ()));
    Sophus::SE3d se3_initial_guess(Sophus::SO3d(init_guess.rotation()), init_guess.translation());

    // ICP Registration using the pipeline (new overload with external initial guess)
    const auto &[frame, keypoints] = odometry_.RegisterFrame(points, timestamps, se3_initial_guess);

    // Get the LiDAR (ICP) pose measurement from the pipeline.
    const auto icp_pose = odometry_.poses().back();
    // Extract 2D measurement from the ICP result.
    double x_meas = icp_pose.translation()(0);
    double y_meas = icp_pose.translation()(1);
    double theta_meas = std::atan2(icp_pose.rotationMatrix()(1,0), icp_pose.rotationMatrix()(0,0));

    // EKF update step using the LiDAR measurement.
    ekf2d_.update(x_meas, y_meas, theta_meas);

    // Get fused (updated) state.
    Eigen::Vector3d fused_state = ekf2d_.getState();
    // Build a 3D pose for publishing.
    Eigen::Isometry3d fused_pose = Eigen::Isometry3d::Identity();
    fused_pose.pretranslate(Eigen::Vector3d(fused_state(0), fused_state(1), 0.0));
    fused_pose.rotate(Eigen::AngleAxisd(fused_state(2), Eigen::Vector3d::UnitZ()));
    const Eigen::Vector3d t_current = fused_pose.translation();
    Eigen::Quaterniond q_current(fused_pose.rotation());

    // Broadcast the transform.
    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = msg.header.stamp;
    transform_msg.header.frame_id = odom_frame_;
    transform_msg.child_frame_id = child_frame_;
    transform_msg.transform.translation.x = t_current.x();
    transform_msg.transform.translation.y = t_current.y();
    transform_msg.transform.translation.z = t_current.z();
    transform_msg.transform.rotation.x = q_current.x();
    transform_msg.transform.rotation.y = q_current.y();
    transform_msg.transform.rotation.z = q_current.z();
    transform_msg.transform.rotation.w = q_current.w();
    tf_broadcaster_->sendTransform(transform_msg);

    // Publish the odometry message.
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = msg.header.stamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = child_frame_;
    odom_msg.pose.pose.position.x = t_current.x();
    odom_msg.pose.pose.position.y = t_current.y();
    odom_msg.pose.pose.position.z = t_current.z();
    odom_msg.pose.pose.orientation.x = q_current.x();
    odom_msg.pose.pose.orientation.y = q_current.y();
    odom_msg.pose.pose.orientation.z = q_current.z();
    odom_msg.pose.pose.orientation.w = q_current.w();
    odom_publisher_->publish(odom_msg);

    // Publish trajectory.
    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header = odom_msg.header;
    pose_msg.pose = odom_msg.pose.pose;
    path_msg_.poses.push_back(pose_msg);
    traj_publisher_->publish(path_msg_);

    // Publish debugging point clouds.
    std_msgs::msg::Header frame_header = msg.header;
    frame_header.frame_id = child_frame_;
    frame_publisher_->publish(kiss_icp_ros::utils::EigenToPointCloud2(frame, frame_header));
    kpoints_publisher_->publish(kiss_icp_ros::utils::EigenToPointCloud2(keypoints, frame_header));

    auto local_map_header = msg.header;
    local_map_header.frame_id = odom_frame_;
    map_publisher_->publish(kiss_icp_ros::utils::EigenToPointCloud2(odometry_.LocalMap(), local_map_header));
}

}  // namespace kiss_icp_ros

//
// -----------------------------------------------------------------------------
// Main Function
// -----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<kiss_icp_ros::OdometryServer>());
    rclcpp::shutdown();
    return 0;
}
