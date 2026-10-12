#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <atomic>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/nailong/solver.hpp"
#include "tasks/nailong/detector.hpp"
#include "tasks/nailong/yolox_detector.hpp"
#include "tools/exiter.hpp"

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/nailong.yaml | 位置参数，yaml配置文件路径 }"
  "{headless       |      | 无窗口模式（覆盖 yaml 里的 show）}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = YAML::LoadFile(config_path);
  bool show = yaml["show"] ? yaml["show"].as<bool>() : true;
  cv::CommandLineParser cli2(argc, argv, keys);
  if (cli2.has("headless")) show = false;

  tools::Exiter exiter;

  // 方向解算器: 像素 -> 云台系/世界系 yaw,pitch
  nailong::Solver solver(config_path);
  // 云台串口 (standard_mpc 协议): 收 GimbalToVision 四元数, 发 VisionToGimbal 角度
  // use_gimbal: false 时 R_gimbal2world = I，ypr_in_world == ypr_in_gimbal
  std::unique_ptr<io::Gimbal> gimbal;
  const bool use_gimbal = yaml["use_gimbal"] ? yaml["use_gimbal"].as<bool>() : false;
  const bool send_to_gimbal = yaml["send_to_gimbal"] ? yaml["send_to_gimbal"].as<bool>() : false;
  if (use_gimbal) gimbal = std::make_unique<io::Gimbal>(config_path);
  if (send_to_gimbal && !use_gimbal) {
    std::cout << "[nailong_vision] send_to_gimbal 需要 use_gimbal: true" << std::endl;
    return 1;
  }

  // 发送线程共享的最新解算结果（世界系角度, rad）
  std::atomic<bool> has_target{false};
  std::atomic<double> send_yaw{0.0}, send_pitch{0.0};
  std::atomic<bool> quit{false};
  std::thread send_thread;
  if (send_to_gimbal) {
    send_thread = std::thread([&]() {
      // 100Hz 下发, 云台侧自己闭环
      auto last = std::chrono::steady_clock::now();
      while (!quit) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last).count() < 0.01) continue;
        last = now;
        if (has_target)
          gimbal->send(true, false, (float)send_yaw.load(), 0, 0, (float)send_pitch.load(), 0, 0);
        else
          gimbal->send(false, false, 0, 0, 0, 0, 0, 0);
      }
    });
  }

  io::Camera camera(config_path);
  auto yaml_all = YAML::LoadFile(config_path);
  const std::string detector_type =
    yaml_all["detector"] ? yaml_all["detector"].as<std::string>() : std::string("yolox");
  // 检测器工厂: yolox = YOLOX 检测 (推荐); grid = ResNet50 网格分类 (兼容旧配置)
  std::unique_ptr<nailong::YoloxDetector> yolox_detector;
  std::unique_ptr<nailong::Detector> grid_detector;
  if (detector_type == "yolox") {
    yolox_detector = std::make_unique<nailong::YoloxDetector>(config_path);
  } else {
    grid_detector = std::make_unique<nailong::Detector>(config_path);
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  double fps = 0;
  double infer_ms = 0;
  bool last_state = false;
  int frame_idx = 0;
  // CPU 上 ResNet50 较重：默认每 2 帧推理一次
  const int infer_every = yaml["infer_every"] ? yaml["infer_every"].as<int>() : 2;
  std::vector<nailong::Detection> last_result;  // 缓存上一次结果

  while (!exiter.exit()) {
    camera.read(img, t);
    if (img.empty()) continue;

    auto t0 = std::chrono::steady_clock::now();

    if (frame_idx % infer_every == 0) {
      last_result =
        yolox_detector ? yolox_detector->detect(img) : grid_detector->detect(img);
      auto t1 = std::chrono::steady_clock::now();
      infer_ms = 0.9 * infer_ms + 0.1 * std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    auto now = std::chrono::steady_clock::now();
    double frame_ms = std::chrono::duration<double, std::milli>(now - t0).count();
    fps = 0.9 * fps + 0.1 * (1000.0 / std::max(frame_ms, 1e-6));

    bool found = !last_result.empty();

    // 解算框中心方向（每帧都用最新 IMU 姿态）
    nailong::Solution sol;
    bool solved = false;
    if (found) {
      try {
        if (use_gimbal && gimbal) {
          // 图像时间戳附近的四元数插值（同 standard_mpc auto_aim_debug_mpc）
          solver.set_R_gimbal2world(gimbal->q(t));
        }
        sol = solver.solve(last_result[0].box);
        solved = true;
        has_target = true;
        send_yaw = sol.yaw_in_world;
        send_pitch = sol.pitch_in_world;
        if (frame_idx % 15 == 0) {
          auto to_deg = [](double rad) { return rad * 180.0 / CV_PI; };
          std::cout << "[solver] yaw_g=" << to_deg(sol.yaw_in_gimbal) << " pitch_g=" << to_deg(sol.pitch_in_gimbal)
                    << " yaw_w=" << to_deg(sol.yaw_in_world) << " pitch_w=" << to_deg(sol.pitch_in_world)
                    << " (deg)" << std::endl;
        }
      } catch (const std::exception & e) {
        std::cout << "[solver] error: " << e.what() << std::endl;
      }
    }

    // 状态变化时打印（headless 模式下也能看）
    if (found != last_state) {
      std::cout << "[nailong_vision] " << (found ? "Nailong!" : "No Nailong") << std::endl;
      last_state = found;
    }
    if (!found) has_target = false;

    if (show) {
      // 画识别框
      for (const auto & d : last_result) {
        cv::rectangle(img, d.box, cv::Scalar(0, 0, 255), 3);
        std::ostringstream label;
        label << "Nailong " << std::fixed << std::setprecision(0) << d.confidence * 100 << "%";
        if (&d == &last_result[0] && solved) {
          auto to_deg = [](double rad) { return rad * 180.0 / CV_PI; };
          label << " yaw:" << std::setprecision(1) << to_deg(sol.yaw_in_gimbal)
                << " pit:" << to_deg(sol.pitch_in_gimbal);
        }
        int base = 0;
        cv::Size text_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &base);
        cv::Point text_org(d.box.x + 5, d.box.y + text_size.height + 8);
        cv::rectangle(img, {text_org.x - 2, text_org.y - text_size.height - 4},
                      {text_org.x + text_size.width + 2, text_org.y + base + 2},
                      cv::Scalar(0, 0, 255), -1);
        cv::putText(img, label.str(), text_org, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 2);
      }

      if (!found) {
        cv::putText(img, "No Nailong", cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 200, 0), 2);
      }

      std::ostringstream info;
      info << "FPS: " << std::fixed << std::setprecision(1) << fps
           << "  infer: " << std::setprecision(0) << infer_ms << "ms";
      cv::putText(img, info.str(), cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(255, 255, 255), 1);

      cv::imshow("nailong_vision", img);
      if ((cv::waitKey(1) & 0xFF) == 'q') break;
    }

    frame_idx++;
  }

  quit = true;
  if (send_thread.joinable()) send_thread.join();
  std::cout << "[nailong_vision] done" << std::endl;
  return 0;
}





