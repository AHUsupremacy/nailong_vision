#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "tools/thread_safe_queue.hpp"

namespace io
{
// 参考 sp_vision 的 io/usbcamera 实现：
// V4L 打开 + 取图线程 + 守护重连线程 + 线程安全队列
class Camera
{
public:
  Camera(const std::string & config_path);
  ~Camera();
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  std::string device_name;

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  std::mutex cap_mutex_;
  cv::VideoCapture cap_;
  std::string open_name_;
  int usb_exposure_, usb_frame_rate_;
  double image_width_, image_height_;
  bool quit_, ok_;
  std::thread capture_thread_;
  std::thread daemon_thread_;
  tools::ThreadSafeQueue<CameraData> queue_;

  void open();
  void close();
};

}  // namespace io

#endif
