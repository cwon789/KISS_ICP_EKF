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

    // 파라미터: 예측 시 yaw 업데이트 여부.
    bool use_yaw_in_prediction = declare_parameter<bool>("use_yaw_in_prediction", true);
    ekf2d_.setUpdateYaw(use_yaw_in_prediction);

    // 파라미터: uncertainty_sigma (초기 추정치 보정을 위한 blending 계수 조절)
    double uncertainty_sigma = declare_parameter<double>("uncertainty_sigma", 0.5);

    // Initialize the KISS-ICP pipeline.
    odometry_ = kiss_icp::pipeline::KissICP(config_);

    // 초기 EKF 업데이트 결과 fallback: 아이덴티티.
    last_ekf_pose_ = Eigen::Isometry3d::Identity();

    // -------------------------------------------------------------------------
    // Subscribers
    // -------------------------------------------------------------------------
    // 휠 odometry는 일반 콜백으로 처리 (동기화 없음)
    wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
         "/bot_sensor/odometer/odometry", rclcpp::SensorDataQoS(),
         std::bind(&OdometryServer::wheelOdometryCallback, this, std::placeholders::_1));

    // LiDAR 메시지 콜백 (업데이트 측정)
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
    if (child_frame_ != "base_link") {
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
         alias_transform_msg.child_frame_id = "base_link";
         br->sendTransform(alias_transform_msg);
    }

    RCLCPP_INFO(get_logger(), "KISS-ICP ROS2 odometry node (with EKF fusion & time sync) initialized");
}

void OdometryServer::wheelOdometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // 예측 단계: 들어오는 휠 odometry마다 EKF 예측 수행.
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

    ekf2d_.predict(v, w, dt);
}

void OdometryServer::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // 업데이트 전에, LiDAR 메시지의 타임스탬프와 마지막 휠 odometry 타임스탬프를 맞춥니다.
    rclcpp::Time lidar_stamp = msg->header.stamp;
    if (lidar_stamp > last_wheel_stamp_) {
         double dt_sync = (lidar_stamp - last_wheel_stamp_).seconds();
         // 추가 움직임 없다고 가정하여 (v=0, w=0) dt_sync만큼 예측 수행.
         ekf2d_.predict(0, 0, dt_sync);
         last_wheel_stamp_ = lidar_stamp;
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

    // 기본 EKF 예측 상태에서 초기 추정치 생성.
    Eigen::Vector3d ekf_state = ekf2d_.getState(); // [x, y, theta]
    Eigen::Isometry3d current_guess = Eigen::Isometry3d::Identity();
    current_guess.pretranslate(Eigen::Vector3d(ekf_state(0), ekf_state(1), 0.0));
    current_guess.rotate(Eigen::AngleAxisd(ekf_state(2), Eigen::Vector3d::UnitZ()));

    // --- 불확실성을 고려한 블렌딩 ---  
    // EKF 공분산 행렬을 이용하여 uncertainty metric 계산.
    Eigen::Matrix3d ekf_cov = ekf2d_.getCovariance();
    double uncertainty_metric = ekf_cov.trace();  // 단순히 trace를 사용 (필요에 따라 다른 metric 사용 가능)
    double sigma = 0.5;  // 파라미터 uncertainty_sigma (디폴트 0.5)
    this->get_parameter("uncertainty_sigma", sigma);
    // 불확실성이 높으면 alpha는 작아지고, 낮으면 alpha는 1에 가까워집니다.
    double alpha = std::exp(-uncertainty_metric / sigma);

    // 만약 이전 업데이트된 EKF 상태가 있다면 (last_ekf_pose_가 유효하다면), 블렌딩.
    // (초기에는 last_ekf_pose_는 아이덴티티)
    Eigen::Vector3d trans_blended = alpha * current_guess.translation() +
                                    (1.0 - alpha) * last_ekf_pose_.translation();
    Eigen::Quaterniond q_current(current_guess.rotation());
    Eigen::Quaterniond q_last(last_ekf_pose_.rotation());
    Eigen::Quaterniond q_blended = q_last.slerp(alpha, q_current);
    Eigen::Isometry3d blended_guess = Eigen::Isometry3d::Identity();
    blended_guess.pretranslate(trans_blended);
    blended_guess.rotate(q_blended);
    // 이 blended_guess를 ICP 초기 추정치로 사용.
    Sophus::SE3d se3_initial_guess(Sophus::SO3d(blended_guess.rotation()), blended_guess.translation());
    // 업데이트 후, 현재 상태를 last_ekf_pose_로 저장.
    last_ekf_pose_ = blended_guess;
    // --- 끝 블렌딩 ---

    // ICP registration 수행 (외부 초기 추정치 사용)
    const auto &[frame, keypoints] = odometry_.RegisterFrame(points, timestamps, se3_initial_guess);

    // ICP 결과 측정치 획득.
    const auto icp_pose = odometry_.poses().back();
    double x_meas = icp_pose.translation()(0);
    double y_meas = icp_pose.translation()(1);
    double theta_meas = std::atan2(icp_pose.rotationMatrix()(1, 0),
                                  icp_pose.rotationMatrix()(0, 0));
    RCLCPP_INFO(get_logger(), "ICP pose: x=%.3f, y=%.3f, theta=%.3f", x_meas, y_meas, theta_meas);

    // EKF 업데이트: ICP 측정치 융합 (이상치는 Mahalanobis 거리로 리젝션).
    ekf2d_.update(x_meas, y_meas, theta_meas);

    // 최종 융합된 상태를 기반으로 포즈 생성 및 퍼블리시.
    Eigen::Vector3d fused_state = ekf2d_.getState();
    Eigen::Isometry3d fused_pose = Eigen::Isometry3d::Identity();
    fused_pose.pretranslate(Eigen::Vector3d(fused_state(0), fused_state(1), 0.0));
    fused_pose.rotate(Eigen::AngleAxisd(fused_state(2), Eigen::Vector3d::UnitZ()));
    const Eigen::Vector3d t_current = fused_pose.translation();
    Eigen::Quaterniond q_current_final(fused_pose.rotation());

    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = msg->header.stamp;
    transform_msg.header.frame_id = odom_frame_;
    transform_msg.child_frame_id = child_frame_;
    transform_msg.transform.translation.x = t_current.x();
    transform_msg.transform.translation.y = t_current.y();
    transform_msg.transform.translation.z = t_current.z();
    transform_msg.transform.rotation.x = q_current_final.x();
    transform_msg.transform.rotation.y = q_current_final.y();
    transform_msg.transform.rotation.z = q_current_final.z();
    transform_msg.transform.rotation.w = q_current_final.w();
    tf_broadcaster_->sendTransform(transform_msg);

    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp = msg->header.stamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = child_frame_;
    odom_msg.pose.pose.position.x = t_current.x();
    odom_msg.pose.pose.position.y = t_current.y();
    odom_msg.pose.pose.position.z = t_current.z();
    odom_msg.pose.pose.orientation.x = q_current_final.x();
    odom_msg.pose.pose.orientation.y = q_current_final.y();
    odom_msg.pose.pose.orientation.z = q_current_final.z();
    odom_msg.pose.pose.orientation.w = q_current_final.w();
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

#include "rclcpp/rclcpp.hpp"
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<kiss_icp_ros::OdometryServer>());
    rclcpp::shutdown();
    return 0;
}
