#ifndef NAILONG__DETECTOR_HPP
#define NAILONG__DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace nailong
{
struct Detection
{
  cv::Rect box;       // 在原图上的识别框
  float confidence;   // 是奶龙的置信度 (0~1)
};

// 奶龙检测器：ResNet50 二分类 (onnx) + 网格定位
// 将画面切成 grid_rows x grid_cols 的网格，逐格分类，
// 判定为奶龙的格子输出识别框
class Detector
{
public:
  Detector(const std::string & config_path);

  std::vector<Detection> detect(const cv::Mat & img);

private:
  cv::dnn::Net net_;
  int input_size_;        // 模型输入尺寸，默认 224
  int grid_rows_;         // 网格行数
  int grid_cols_;         // 网格列数
  float conf_threshold_;  // 置信度阈值
  bool initialized_ = false;
};

}  // namespace nailong

#endif
