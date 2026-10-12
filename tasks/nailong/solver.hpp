#ifndef NAILONG__SOLVER_HPP
#define NAILONG__SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>

#include "detector.hpp"

namespace nailong
{
// 单个目标的解算结果
// 奶龙跟踪只需要方向（yaw/pitch），不需要距离，因此 xyz 是单位方向向量
struct Solution
{
  Eigen::Vector3d xyz_in_camera;  // 相机坐标系下的单位方向
  Eigen::Vector3d xyz_in_gimbal;  // 云台坐标系下的单位方向
  Eigen::Vector3d xyz_in_world;   // 世界坐标系下的单位方向（经 IMU 四元数变换）
  double yaw_in_gimbal;           // rad
  double pitch_in_gimbal;         // rad
  double yaw_in_world;            // rad
  double pitch_in_world;          // rad
};

// 方向解算器：像素框中心 -> 相机系方向 -> 云台系 -> 世界系
// 参考 sp_vision tasks/auto_aim/solver 的坐标链路，去掉 PnP（无已知尺寸）
class Solver
{
public:
  explicit Solver(const std::string & config_path);

  // 由 IMU 四元数更新 云台->世界 旋转（参考 auto_aim::Solver::set_R_gimbal2world）
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  // box: 检测框（已外扩），返回该方向的解算结果
  Solution solve(const cv::Rect & box) const;

  Eigen::Matrix3d R_gimbal2world() const { return R_gimbal2world_; }

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;
};

}  // namespace nailong

#endif  // NAILONG__SOLVER_HPP
