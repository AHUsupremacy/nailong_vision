#ifndef NAILONG__YOLOX_DETECTOR_HPP
#define NAILONG__YOLOX_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "detector.hpp"

namespace nailong
{
// 奶龙检测器（YOLOX 版）：直接输出检测框
// 模型: nkxingxh/NailongDetection nailong_v2.6_tiny.onnx (AGPL-3.0)
// 输入: BGR 0-255 (该 ONNX 导出未经归一化), 416x416, letterbox
// 输出: (1, 3549, 13) 半解码格式, 需按 YOLOX 规则补 grid + stride
class YoloxDetector
{
public:
  YoloxDetector(const std::string & config_path);

  std::vector<Detection> detect(const cv::Mat & img);

private:
  cv::dnn::Net net_;
  int input_size_;        // 416 (tiny)
  float conf_threshold_;  // 置信度阈值
  float nms_threshold_;
  float expand_ratio_;   // 框扩展系数（模型只框脸部核心，>1 外扩包住整个头）
  int nailong_class_id_;  // labels.txt 里 nailong = 1
  // 预生成的 anchor grid: (grid_x, grid_y, stride) 按输出行序
  cv::Mat grids_;         // CV_32F, N x 3
  bool initialized_ = false;
};

}  // namespace nailong

#endif
