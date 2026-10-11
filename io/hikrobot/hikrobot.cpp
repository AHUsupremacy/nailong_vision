#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include <iostream>
#include <unordered_map>

using namespace std::chrono_literals;

namespace io
{
HikRobot::HikRobot(double exposure_ms, double gain, const std::string & vid_pid)
: exposure_us_(exposure_ms * 1e3), gain_(gain), queue_(1), daemon_quit_(false), vid_(-1), pid_(-1)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) std::cout << "[nailong_vision] Unable to init libusb!" << std::endl;

  daemon_thread_ = std::thread{[this] {
    std::cout << "[nailong_vision] HikRobot's daemon thread started." << std::endl;

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    std::cout << "[nailong_vision] HikRobot's daemon thread stopped." << std::endl;
  }};
}

HikRobot::~HikRobot()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  std::cout << "[nailong_vision] HikRobot destructed." << std::endl;
}

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

void HikRobot::capture_start()
{
  capturing_ = false;
  capture_quit_ = false;

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_EnumDevices failed: 0x" << std::hex << ret << std::dec
              << std::endl;
    return;
  }

  if (device_list.nDeviceNum == 0) {
    std::cout << "[nailong_vision] Not found camera!" << std::endl;
    return;
  }

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_CreateHandle failed: 0x" << std::hex << ret << std::dec
              << std::endl;
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_OpenDevice failed: 0x" << std::hex << ret << std::dec
              << std::endl;
    return;
  }

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, 150);

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_StartGrabbing failed: 0x" << std::hex << ret
              << std::dec << std::endl;
    return;
  }

  capture_thread_ = std::thread{[this] {
    std::cout << "[nailong_vision] HikRobot's capture thread started." << std::endl;

    capturing_ = true;

    MV_FRAME_OUT raw;
    MV_CC_PIXEL_CONVERT_PARAM cvt_param;

    while (!capture_quit_) {
      std::this_thread::sleep_for(1ms);

      unsigned int ret;
      unsigned int nMsec = 100;

      ret = MV_CC_GetImageBuffer(handle_, &raw, nMsec);
      if (ret != MV_OK) {
        std::cout << "[nailong_vision] MV_CC_GetImageBuffer failed: 0x" << std::hex << ret
                  << std::dec << std::endl;
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      cvt_param.nWidth = raw.stFrameInfo.nWidth;
      cvt_param.nHeight = raw.stFrameInfo.nHeight;

      cvt_param.pSrcData = raw.pBufAddr;
      cvt_param.nSrcDataLen = raw.stFrameInfo.nFrameLen;
      cvt_param.enSrcPixelType = raw.stFrameInfo.enPixelType;

      cvt_param.pDstBuffer = img.data;
      cvt_param.nDstBufferSize = img.total() * img.elemSize();
      cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

      // 参考 sp_vision：Bayer 转彩色（MV-CS016-10UC 出 BayerRG8）
      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      cv::cvtColor(img, dst_image, type_map.at(pixel_type));
      img = dst_image;

      queue_.push({img, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        std::cout << "[nailong_vision] MV_CC_FreeImageBuffer failed: 0x" << std::hex << ret
                  << std::dec << std::endl;
        break;
      }
    }

    capturing_ = false;
    std::cout << "[nailong_vision] HikRobot's capture thread stopped." << std::endl;
  }};
}

void HikRobot::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_StopGrabbing failed: 0x" << std::hex << ret << std::dec
              << std::endl;
    return;
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_CloseDevice failed: 0x" << std::hex << ret << std::dec
              << std::endl;
    return;
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_DestroyHandle failed: 0x" << std::hex << ret
              << std::dec << std::endl;
    return;
  }
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_SetFloatValue(\"" << name << "\", " << value
              << ") failed: 0x" << std::hex << ret << std::dec << std::endl;
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    std::cout << "[nailong_vision] MV_CC_SetEnumValue(\"" << name << "\", " << value
              << ") failed: 0x" << std::hex << ret << std::dec << std::endl;
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    std::cout << "[nailong_vision] Invalid vid_pid: \"" << vid_pid << "\"" << std::endl;
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    std::cout << "[nailong_vision] Invalid vid_pid: \"" << vid_pid << "\"" << std::endl;
  }
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    std::cout << "[nailong_vision] Unable to open usb!" << std::endl;
    return;
  }

  if (libusb_reset_device(handle))
    std::cout << "[nailong_vision] Unable to reset usb!" << std::endl;
  else
    std::cout << "[nailong_vision] Reset usb successfully :)" << std::endl;

  libusb_close(handle);
}

}  // namespace io
