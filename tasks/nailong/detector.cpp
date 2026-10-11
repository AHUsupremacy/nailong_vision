#include "detector.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

namespace nailong
{
Detector::Detector(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto model_path = yaml["model_path"].as<std::string>();
  input_size_ = yaml["input_size"] ? yaml["input_size"].as<int>() : 224;
  grid_rows_ = yaml["grid_rows"] ? yaml["grid_rows"].as<int>() : 2;
  grid_cols_ = yaml["grid_cols"] ? yaml["grid_cols"].as<int>() : 2;
  conf_threshold_ = yaml["min_confidence"] ? yaml["min_confidence"].as<float>() : 0.8f;

  net_ = cv::dnn::readNetFromONNX(model_path);
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  initialized_ = true;
}

std::vector<Detection> Detector::detect(const cv::Mat & img)
{
  std::vector<Detection> detections;
  if (!initialized_ || img.empty()) return detections;

  const int img_w = img.cols;
  const int img_h = img.rows;
  const int cell_w = img_w / grid_cols_;
  const int cell_h = img_h / grid_rows_;

  // 按网格收集 cells
  std::vector<cv::Rect> cells;
  std::vector<cv::Mat> cell_imgs;
  cells.reserve(grid_rows_ * grid_cols_);
  cell_imgs.reserve(grid_rows_ * grid_cols_);

  for (int r = 0; r < grid_rows_; r++) {
    for (int c = 0; c < grid_cols_; c++) {
      cv::Rect rect(c * cell_w, r * cell_h, cell_w, cell_h);
      rect &= cv::Rect(0, 0, img_w, img_h);
      if (rect.empty()) continue;
      cells.push_back(rect);
      cell_imgs.push_back(img(rect));
    }
  }

  if (cells.empty()) return detections;

  // 预处理与训练一致：Resize(224,224) + ToTensor（无归一化，RGB 顺序）
  cv::Mat blob = cv::dnn::blobFromImages(
    cell_imgs, 1.0 / 255.0, cv::Size(input_size_, input_size_), cv::Scalar(), true, false, CV_32F);

  net_.setInput(blob);
  cv::Mat output = net_.forward();  // shape: N x 2

  for (size_t i = 0; i < cells.size(); i++) {
    cv::Mat logits = output.row(i);
    double max_val;
    cv::Point max_loc;
    cv::minMaxLoc(logits, nullptr, &max_val, nullptr, &max_loc);

    // softmax 置信度（减 max 防溢出）
    float l0 = logits.at<float>(0);
    float l1 = logits.at<float>(1);
    float m = std::max(l0, l1);
    double e0 = std::exp(l0 - m);
    double e1 = std::exp(l1 - m);
    double conf = e1 / (e0 + e1);

    // class 1 = 奶龙（与 easy_inference.py 一致）
    if (max_loc.x == 1 && conf >= conf_threshold_) {
      detections.push_back({cells[i], static_cast<float>(conf)});
    }
  }

  return detections;
}

}  // namespace nailong



