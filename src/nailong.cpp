#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "tasks/nailong/detector.hpp"
#include "tools/exiter.hpp"

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/nailong.yaml | 位置参数，yaml配置文件路径 }";

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

  tools::Exiter exiter;

  io::Camera camera(config_path);
  nailong::Detector detector(config_path);

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
      last_result = detector.detect(img);
      auto t1 = std::chrono::steady_clock::now();
      infer_ms = 0.9 * infer_ms + 0.1 * std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    auto now = std::chrono::steady_clock::now();
    double frame_ms = std::chrono::duration<double, std::milli>(now - t0).count();
    fps = 0.9 * fps + 0.1 * (1000.0 / std::max(frame_ms, 1e-6));

    bool found = !last_result.empty();

    // 状态变化时打印（headless 模式下也能看）
    if (found != last_state) {
      std::cout << "[nailong_vision] " << (found ? "Nailong!" : "No Nailong") << std::endl;
      last_state = found;
    }

    if (show) {
      // 画识别框
      for (const auto & d : last_result) {
        cv::rectangle(img, d.box, cv::Scalar(0, 0, 255), 3);
        std::ostringstream label;
        label << "Nailong " << std::fixed << std::setprecision(0) << d.confidence * 100 << "%";
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

  std::cout << "[nailong_vision] done" << std::endl;
  return 0;
}



