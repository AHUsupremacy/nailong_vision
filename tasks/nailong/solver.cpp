#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>

namespace nailong
{
static Eigen::Vector3d xyz2yp(const Eigen::Vector3d & xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];
  auto yaw = std::atan2(y, x);
  auto pitch = std::atan2(z, std::sqrt(x * x + y * y));
  return {yaw, pitch, 1.0};
}

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  // 相机内参/畸变（标定后填入 yaml，格式同 sp_vision standard3.yaml）
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K(camera_matrix_data.data());
  cv::eigen2cv(K, camera_matrix_);
  distort_coeffs_ = cv::Mat(1, distort_coeffs_data.size(), CV_64F);
  for (size_t i = 0; i < distort_coeffs_data.size(); i++)
    distort_coeffs_.at<double>(0, i) = distort_coeffs_data[i];

  // 云台<->IMUbody、相机<->云台 外参（格式同 standard3.yaml）
  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Vector3d(t_camera2gimbal_data.data());
}

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  // 与 auto_aim 相同: 先把 IMU 的姿态变换到云台坐标系约定下
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

Solution Solver::solve(const cv::Rect & box) const
{
  Solution sol;

  // 1. 框中心像素 -> 归一化相机系方向（去畸变，无 P2 矩阵 => 单位射线）
  cv::Point2f center(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
  std::vector<cv::Point2f> pts{center}, rays;
  cv::undistortPoints(pts, rays, camera_matrix_, distort_coeffs_);
  Eigen::Vector3d ray(rays[0].x, rays[0].y, 1.0);
  ray.normalize();
  sol.xyz_in_camera = ray;

  // 2. 相机系 -> 云台系（平移量对方向影响很小，保留以与 auto_aim 一致）
  sol.xyz_in_gimbal = R_camera2gimbal_ * ray + t_camera2gimbal_;
  sol.xyz_in_gimbal.normalize();

  // 3. 云台系 -> 世界系（经 IMU 四元数）
  sol.xyz_in_world = R_gimbal2world_ * sol.xyz_in_gimbal;

  // 4. 方向 -> yaw/pitch（弧度）
  Eigen::Vector3d yp_gimbal = xyz2yp(sol.xyz_in_gimbal);
  Eigen::Vector3d yp_world = xyz2yp(sol.xyz_in_world);
  sol.yaw_in_gimbal = yp_gimbal[0];
  sol.pitch_in_gimbal = yp_gimbal[1];
  sol.yaw_in_world = yp_world[0];
  sol.pitch_in_world = yp_world[1];
  return sol;
}

}  // namespace nailong
