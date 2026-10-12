// 视频检测测试: ./build/video_test <视频路径> [配置文件路径]
// 逐帧跑 YOLOX 检测 + 方向解算, 画框显示, 显示实际 FPS / 检出率
// q 或 Ctrl+C 退出
#include <algorithm>
#include <filesystem>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "tasks/nailong/detector.hpp"
#include "tasks/nailong/solver.hpp"
#include "tasks/nailong/yolox_detector.hpp"

int main(int argc, char * argv[])
{
  // 无参数: 默认跑 test_videos/ 下所有视频; 有参数: 只跑指定的那个
  std::vector<std::string> videos;
  if (argc >= 2) {
    videos.push_back(argv[1]);
  } else {
    for (const auto & entry : std::filesystem::directory_iterator("test_videos"))
      if (entry.is_regular_file()) videos.push_back(entry.path().string());
    std::sort(videos.begin(), videos.end());
    if (videos.empty()) {
      std::cout << "test_videos/ 下没有视频, 用法: video_test <视频路径>" << std::endl;
      return 0;
    }
  }
  auto config_path = argc > 2 ? argv[2] : "configs/nailong.yaml";

  for (const auto & video_path : videos) {
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cout << "无法打开视频: " << video_path << std::endl;
    continue;
  }
  const bool headless = std::getenv("NAILONG_HEADLESS") != nullptr;
  double video_fps = cap.get(cv::CAP_PROP_FPS);
  if (video_fps <= 0 || video_fps > 240) video_fps = 30;
  std::cout << "视频: " << video_path << " " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
            << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << " " << video_fps << "fps" << std::endl;

  nailong::YoloxDetector detector(config_path);
  nailong::Solver solver(config_path);

  cv::Mat img;
  double fps = 0;
  int frame_idx = 0, hit_count = 0;
  const int infer_every = 1;  // 离线测试每帧都推理

  while (true) {
    auto t0 = std::chrono::steady_clock::now();
    cap >> img;
    if (img.empty()) {
      std::cout << "播放结束" << std::endl;
      break;
    }

    auto dets = detector.detect(img);
    if (!dets.empty()) {
      hit_count++;
      // 解算第一个框的方向
      try {
        auto sol = solver.solve(dets[0].box);
        auto to_deg = [](double rad) { return rad * 180.0 / CV_PI; };
        if (frame_idx % 30 == 0)
          std::cout << "[solver] yaw=" << to_deg(sol.yaw_in_gimbal)
                    << " pitch=" << to_deg(sol.pitch_in_gimbal) << " (deg)" << std::endl;
      } catch (const std::exception & e) {
        std::cout << "[solver] error: " << e.what() << std::endl;
      }
    }

    // 画框
    for (size_t i = 0; i < dets.size(); i++) {
      const auto & d = dets[i];
      cv::rectangle(img, d.box, cv::Scalar(0, 0, 255), 3);
      std::ostringstream label;
      label << "Nailong " << std::fixed << std::setprecision(0) << d.confidence * 100 << "%";
      if (i == 0) {
        // 标出框中心（跟踪点）
        cv::Point c(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
        cv::drawMarker(img, c, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);
        label << " yaw/pit";
      }
      int base = 0;
      cv::Size ts = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &base);
      cv::Point org(d.box.x + 5, d.box.y + ts.height + 8);
      cv::rectangle(img, {org.x - 2, org.y - ts.height - 4}, {org.x + ts.width + 2, org.y + base + 2},
                    cv::Scalar(0, 0, 255), -1);
      cv::putText(img, label.str(), org, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fps = 0.9 * fps + 0.1 * (1000.0 / std::max(ms, 1e-6));

    std::ostringstream info;
    info << "FPS: " << std::fixed << std::setprecision(1) << fps << "  hit: " << std::setprecision(1)
         << 100.0 * hit_count / (frame_idx + 1) << "%";
    cv::putText(img, info.str(), {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

    if (!headless) {
      cv::imshow("nailong video test", img);
      if ((cv::waitKey(1) & 0xFF) == 'q') break;
    }
    frame_idx++;
  }

  std::cout << "总帧数: " << frame_idx + 1 << "  检出帧: " << hit_count << std::endl;
  cv::destroyAllWindows();
  }  // for videos
  return 0;
}
