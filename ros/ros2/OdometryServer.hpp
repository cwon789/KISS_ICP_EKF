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
// Simple 2D EKF class (state: [x, y, theta]) with Mahalanobis outlier rejection
// 그리고 예측 시 yaw 업데이트 여부를 제어하는 옵션을 포함합니다.
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
    Q_(0,0) = 0.01 / 100.0 ;
    Q_(1,1) = 0.01 / 100.0 ;
    Q_(2,2) = 0.01 / 100.0 ;

    // Measurement noise R
    R_.setZero();
    R_(0,0) = 0.02;
    R_(1,1) = 0.02;
    R_(2,2) = 0.02;

    // Mahalanobis threshold (예: 3 DOF에서 보수적으로 9.21)
    mahalanobis_threshold_ = 9.21;

    // 기본적으로 yaw 업데이트를 수행 (true)
    update_yaw_ = true;
  }

  Eigen::Vector3d getState() const { return state_; }
  
  // 공분산 행렬 반환 (초기 추정의 불확실성을 평가하기 위함)
  Eigen::Matrix3d getCovariance() const { return P_; }

  // yaw 업데이트 여부를 설정 (true: yaw 업데이트, false: yaw 유지)
  void setUpdateYaw(bool flag) { update_yaw_ = flag; }

  // Predict: 휠 odometry (v, w, dt)를 통합하여 상태를 예측.
  // update_yaw_가 false이면 yaw는 업데이트하지 않습니다.
  void predict(double v, double w, double dt)
  {
    double x = state_(0);
    double y = state_(1);
    double theta = state_(2);

    double new_x = x + v * std::cos(theta) * dt;
    double new_y = y + v * std::sin(theta) * dt;
    double new_theta = update_yaw_ ? (theta + w * dt) : theta;

    state_ << new_x, new_y, new_theta;

    // Jacobian 계산.
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    if (update_yaw_) {
      F(0,2) = -v * std::sin(theta) * dt;
      F(1,2) =  v * std::cos(theta) * dt;
    } else {
      F(0,2) = 0;
      F(1,2) = 0;
    }
    P_ = F * P_ * F.transpose() + Q_;
  }

  // Update: 측정치 (x, y, theta)를 융합. Mahalanobis 거리를 이용해 이상치(reject) 여부 판단.
  void update(double x_meas, double y_meas, double theta_meas)
  {
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    Eigen::Vector3d z(x_meas, y_meas, theta_meas);
    Eigen::Vector3d z_pred = H * state_;

    Eigen::Matrix3d S = H * P_ * H.transpose() + R_;
    Eigen::Vector3d r = z - z_pred;
    double mahalanobis = r.transpose() * S.inverse() * r;
    if (mahalanobis > mahalanobis_threshold_) {
      RCLCPP_WARN(rclcpp::get_logger("EKF2D"), "Measurement rejected (Mahalanobis distance = %.3f)", mahalanobis);
      return;
    }
    Eigen::Matrix3d K = P_ * H.transpose() * S.inverse();
    state_ = state_ + K * r;
    P_ = (Eigen::Matrix3d::Identity() - K * H) * P_;
  }

private:
  Eigen::Vector3d state_; // [x, y, theta]
  Eigen::Matrix3d P_;
  Eigen::Matrix3d Q_;
  Eigen::Matrix3d R_;
  double mahalanobis_threshold_;
  bool update_yaw_; // true: update yaw in prediction, false: keep yaw unchanged.
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
    std::string odom_frame_{"odom"};
    std::string child_frame_{"base_link"};

    // EKF filter for 2D state.
    EKF2D ekf2d_;

    // Last wheel odometry timestamp.
    rclcpp::Time last_wheel_stamp_;

    // 이전 EKF 업데이트 결과를 저장 (uncertainty 보정을 위한 fallback)
    Eigen::Isometry3d last_ekf_pose_;
};

}  // namespace kiss_icp_ros
