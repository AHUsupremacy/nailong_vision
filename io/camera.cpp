#include "camera.hpp"

#include <iostream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

#include "hikrobot/hikrobot.hpp"
#include "usbcamera.hpp"

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto camera_name = yaml["camera_name"].as<std::string>();

  if (camera_name == "hikrobot") {
    auto exposure_ms = yaml["exposure_ms"] ? yaml["exposure_ms"].as<double>() : 2.0;
    auto gain = yaml["gain"] ? yaml["gain"].as<double>() : 16.0;
    auto vid_pid =
      yaml["vid_pid"] ? yaml["vid_pid"].as<std::string>() : std::string("2bdf:0001");
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain, vid_pid);
  }

  else if (camera_name == "video0") {  // USB 相机：camera_name 填 /dev/videoX 的 X
    camera_ = std::make_unique<UsbCamera>(config_path);
  }

  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);
}

}  // namespace io
