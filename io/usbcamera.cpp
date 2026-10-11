#include "usbcamera.hpp"

#include <iostream>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

namespace io
{
UsbCamera::UsbCamera(const std::string & config_path)
: quit_(false), ok_(false), queue_(1)
{
  auto yaml = YAML::LoadFile(config_path);
  open_name_ = yaml["camera_name"].as<std::string>();
  image_width_ = yaml["image_width"].as<double>();
  image_height_ = yaml["image_height"].as<double>();
  usb_exposure_ = yaml["usb_exposure"].as<int>();
  usb_frame_rate_ = yaml["usb_frame_rate"].as<int>();

  open();

  // 守护线程：断线自动重连（参考 sp_vision io/usbcamera）
  daemon_thread_ = std::thread{[this] {
    while (!quit_) {
      std::this_thread::sleep_for(100ms);

      if (ok_) continue;

      if (capture_thread_.joinable()) capture_thread_.join();

      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        close();
      }

      std::cout << "[nailong_vision] trying to reopen " << open_name_ << std::endl;
      open();
    }
  }};
}

UsbCamera::~UsbCamera()
{
  quit_ = true;
  {
    std::lock_guard<std::mutex> lock(cap_mutex_);
    close();
  }
  if (daemon_thread_.joinable()) daemon_thread_.join();
  if (capture_thread_.joinable()) capture_thread_.join();
}

void UsbCamera::open()
{
  std::lock_guard<std::mutex> lock(cap_mutex_);
  std::string true_device_name = "/dev/" + open_name_;
  cap_.open(true_device_name, cv::CAP_V4L);
  if (!cap_.isOpened()) {
    std::cout << "[nailong_vision] failed to open " << true_device_name << std::endl;
    return;
  }

  cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap_.set(cv::CAP_PROP_FPS, usb_frame_rate_);
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
  cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);
  cap_.set(cv::CAP_PROP_EXPOSURE, usb_exposure_);

  device_name = open_name_;
  std::cout << "[nailong_vision] " << device_name << " opened "
            << static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)) << "x"
            << static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT))
            << " fps:" << cap_.get(cv::CAP_PROP_FPS) << std::endl;

  // 取图线程
  capture_thread_ = std::thread{[this] {
    ok_ = true;
    std::this_thread::sleep_for(50ms);
    while (!quit_) {
      cv::Mat img;
      bool success;
      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        if (!cap_.isOpened()) break;
        success = cap_.read(img);
      }

      if (!success) {
        std::cout << "[nailong_vision] failed to read frame, exiting capture thread" << std::endl;
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      queue_.push({img, timestamp});
    }
    ok_ = false;
  }};
}

void UsbCamera::close()
{
  if (cap_.isOpened()) {
    cap_.release();
    std::cout << "[nailong_vision] camera released" << std::endl;
  }
}

void UsbCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);
  img = data.img;
  timestamp = data.timestamp;
}

}  // namespace io
