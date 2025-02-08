// MIT License
//
// [License text omitted for brevity]

#include "OdometryServer.hpp"
#include "Utils.hpp"  // Utility functions for point cloud conversion

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

namespace kiss_icp_ros {

OdometryServer::OdometryServer() 
  : rclcpp::Node("odometry_node"),
    odometry_(config_)
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
    // 예측 (휠 odometry)는 일반 콜백으로 처리 (동기화 없음)
    wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
         "/bot_sensor/odometer/odometry", rclcpp::SensorDataQoS(),
         std::bind(&OdometryServer::wheelOdometryCallback, this, std::placeholders::_1));

    // 업데이트 (ICP)는 LiDAR 메시지 콜백에서 처리
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
         "/bot_sensor/converted/laser_scan", rclcpp::SensorDataQoS(),
         std::bind(&OdometryServer::lidarCallback, this, std::placeholders::_1));

    // -------------------------------------------------------------------------
    // Publishers
    // -------------------------------------------------------------------------
    rclcpp::QoS qos(rclcpp::KeepLast{queue_size_});
    odom_publisher_    = create_publisher<nav_msgs::msg::Odometry>("odometry", qos);
    frame_publisher_   = create_publisher<sensor_msgs::msg::PointCloud2>("frame", qos);
    kpoints_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("keypoints", qos);
    map_publisher_     = create_publisher<sensor_msgs::msg::PointCloud2>("local_map", qos);
    tf_broadcaster_    = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    path_msg_.header.frame_id = odom_frame_;
    traj_publisher_    = create_publisher<nav_msgs::msg::Path>("trajectory", qos);

    // -------------------------------------------------------------------------
    // Broadcast static transform if needed.
    // -------------------------------------------------------------------------
    if (child_frame_ != "base_link_LW") {
         auto br = std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
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
         alias_transform_msg.child_frame_id = "base_link_LW";
         br->sendTransform(alias_transform_msg);
    }

    RCLCPP_INFO(get_logger(), "KISS-ICP ROS2 odometry node (with EKF fusion & time sync) initialized");
}

void OdometryServer::wheelOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // 예측 단계에서는 동기화 없이 들어오는 휠 odometry 메시지마다 EKF 예측 수행.
    double v = msg->twist.twist.linear.x;    // [m/s]
    double w = msg->twist.twist.angular.z;     // [rad/s]
    rclcpp::Time stamp = msg->header.stamp;

    double dt = 0.0;
    if (last_wheel_stamp_.nanoseconds() != 0) {
         dt = (stamp - last_wheel_stamp_).seconds();
         if (dt < 0.0) {
              RCLCPP_WARN(get_logger(), "Negative dt in wheel odometry, setting dt=0");
              dt = 0.0;
         }
    }
    last_wheel_stamp_ = stamp;

    // EKF 예측.
    ekf2d_.predict(v, w, dt);
}

void OdometryServer::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 측정 업데이트 전에, LiDAR 메시지의 타임스탬프에 맞추어 예측 상태를 보정합니다.
    rclcpp::Time lidar_stamp = msg->header.stamp;
    if (lidar_stamp > last_wheel_stamp_) {
         double dt_sync = (lidar_stamp - last_wheel_stamp_).seconds();
         // 추가 움직임이 없는 것으로 가정하고, dt_sync만큼 EKF 상태를 전진(0,0 입력)합니다.
         ekf2d_.predict(0, 0, dt_sync);
         last_wheel_stamp_ = lidar_stamp;  // 동기화 후 타임스탬프 갱신.
    }
    
    // LiDAR 데이터 처리: 포인트 클라우드를 Eigen 형식으로 변환 및 extrinsic 적용.
    const auto &msg_ref = *msg;
    auto points = kiss_icp_ros::utils::PointCloud2ToEigen(msg_ref);
    const auto timestamps = [&]() -> std::vector<double> {
         if (!config_.deskew)
              return {};
         return kiss_icp_ros::utils::GetTimestamps(msg_ref);
    }();

    // --- 하드코딩된 extrinsic (translation 및 yaw 적용) ---
    double extrinsic_tx = 0.7237;
    double extrinsic_ty = 0.0;
    double extrinsic_theta = 0.0;
    Eigen::Isometry3d extrinsic = Eigen::Isometry3d::Identity();
    extrinsic.pretranslate(Eigen::Vector3d(extrinsic_tx, extrinsic_ty, 0.0));
    extrinsic.rotate(Eigen::AngleAxisd(extrinsic_theta, Eigen::Vector3d::UnitZ()));
    for (auto &pt : points) {
         pt = extrinsic * pt;
    }
    // --------------------------------------------------------------

    // 업데이트 단계에서는 EKF 예측 상태를 기반으로 초기 추정치 구성.
    Eigen::Vector3d ekf_state = ekf2d_.getState(); // [x, y, theta]
    Eigen::Isometry3d init_guess = Eigen::Isometry3d::Identity();
    init_guess.pretranslate(Eigen::Vector3d(ekf_state(0), ekf_state(1), 0.0));
    init_guess.rotate(Eigen::AngleAxisd(ekf_state(2), Eigen::Vector3d::UnitZ()));
    Sophus::SE3d se3_initial_guess(Sophus::SO3d(init_guess.rotation()), init_guess.translation());

    // ICP registration 수행 (외부 초기 추정치 사용)
    const auto &[frame, keypoints] = odometry_.RegisterFrame(points, timestamps, se3_initial_guess);

    // ICP 결과로부터 측정값 획득.
    const auto icp_pose = odometry_.poses().back();
    double x_meas = icp_pose.translation()(0);
    double y_meas = icp_pose.translation()(1);
    double theta_meas = std::atan2(icp_pose.rotationMatrix()(1, 0),
                                  icp_pose.rotationMatrix()(0, 0));

    // EKF 업데이트: ICP 측정값을 융합하여 최종 상태 갱신.
    ekf2d_.update(x_meas, y_meas, theta_meas);

    // 최종 융합된 상태를 기반으로 포즈 생성 및 퍼블리시.
    Eigen::Vector3d fused_state = ekf2d_.getState();
    Eigen::Isometry3d fused_pose = Eigen::Isometry3d::Identity();
    fused_pose.pretranslate(Eigen::Vector3d(fused_state(0), fused_state(1), 0.0));
    fused_pose.rotate(Eigen::AngleAxisd(fused_state(2), Eigen::Vector3d::UnitZ()));
    const Eigen::Vector3d t_current = fused_pose.translation();
    Eigen::Quaterniond q_current(fused_pose.rotation());

    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = msg->header.stamp;
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

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = msg->header.stamp;
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

    geometry_msgs::msg::PoseStamped pose_msg;
    pose_msg.header = odom_msg.header;
    pose_msg.pose = odom_msg.pose.pose;
    path_msg_.poses.push_back(pose_msg);
    traj_publisher_->publish(path_msg_);

    // 디버깅용 점군 퍼블리싱.
    std_msgs::msg::Header frame_header = msg->header;
    frame_header.frame_id = child_frame_;
    frame_publisher_->publish(kiss_icp_ros::utils::EigenToPointCloud2(frame, frame_header));
    kpoints_publisher_->publish(kiss_icp_ros::utils::EigenToPointCloud2(keypoints, frame_header));

    auto local_map_header = msg->header;
    local_map_header.frame_id = odom_frame_;
    map_publisher_->publish(kiss_icp_ros::utils::EigenToPointCloud2(odometry_.LocalMap(), local_map_header));
}

}  // namespace kiss_icp_ros

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<kiss_icp_ros::OdometryServer>());
    rclcpp::shutdown();
    return 0;
}
