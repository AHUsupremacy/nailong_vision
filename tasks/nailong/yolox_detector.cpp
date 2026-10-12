#include "yolox_detector.hpp"

#include <yaml-cpp/yaml.h>

namespace nailong
{
YoloxDetector::YoloxDetector(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto model_path = yaml["yolox_model_path"]
                      ? yaml["yolox_model_path"].as<std::string>()
                      : std::string("model/nailong_v2.6_tiny.onnx");
  input_size_ = yaml["yolox_input_size"] ? yaml["yolox_input_size"].as<int>() : 416;
  conf_threshold_ = yaml["yolox_confidence"] ? yaml["yolox_confidence"].as<float>() : 0.3f;
  nms_threshold_ = yaml["yolox_nms"] ? yaml["yolox_nms"].as<float>() : 0.45f;
  nailong_class_id_ = yaml["yolox_class_id"] ? yaml["yolox_class_id"].as<int>() : 1;

  // 预生成 anchor grid：strides [8,16,32]，每块行序 y*ws+x
  std::vector<cv::Vec3f> grids;
  const int strides[] = {8, 16, 32};
  for (int s : strides) {
    int ws = input_size_ / s;
    for (int gy = 0; gy < ws; gy++)
      for (int gx = 0; gx < ws; gx++) grids.push_back({static_cast<float>(gx), static_cast<float>(gy), static_cast<float>(s)});
  }
  grids_ = cv::Mat(static_cast<int>(grids.size()), 3, CV_32F, grids.data()).clone();

  net_ = cv::dnn::readNetFromONNX(model_path);
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  initialized_ = true;
}

std::vector<Detection> YoloxDetector::detect(const cv::Mat & img)
{
  std::vector<Detection> detections;
  if (!initialized_ || img.empty()) return detections;

  // ---- letterbox 到 416x416（保持纵横比，114 填充）----
  const float r = std::min(static_cast<float>(input_size_) / img.cols,
                           static_cast<float>(input_size_) / img.rows);
  const int nw = cvRound(img.cols * r), nh = cvRound(img.rows * r);
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(nw, nh));
  cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
  const int dx = (input_size_ - nw) / 2, dy = (input_size_ - nh) / 2;
  resized.copyTo(canvas(cv::Rect(dx, dy, nw, nh)));

  // ---- 推理（注意: 该模型期望 BGR 0-255, 不做 /255 归一化）----
  cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0, cv::Size(input_size_, input_size_),
                                        cv::Scalar(), false, false, CV_32F);
  net_.setInput(blob);
  cv::Mat output = net_.forward();  // 1 x N x 13
  cv::Mat preds = output.reshape(1, output.size[1]);  // N x 13

  const int n = preds.rows;
  cv::Mat obj = preds.col(4);
  cv::Mat cls_scores = preds.colRange(5, preds.cols);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i = 0; i < n; i++) {
    float o = obj.at<float>(i);
    if (o < conf_threshold_) continue;

    cv::Point max_loc;
    double max_cls;
    cv::minMaxLoc(cls_scores.row(i), nullptr, &max_cls, nullptr, &max_loc);
    float score = o * static_cast<float>(max_cls);
    if (score < conf_threshold_) continue;

    int class_id = max_loc.x;
    // 跟踪只需要奶龙这一类（其余类是该项目自加的，忽略）
    if (class_id != nailong_class_id_) continue;

    // YOLOX 解码: xy = (sig*2-0.5 + grid) * stride (输出已是 sig*2-0.5)
    //             wh = (sig*2)^2 * stride      (输出已是 (sig*2)^2)
    float gx = grids_.at<float>(i, 0), gy = grids_.at<float>(i, 1), s = grids_.at<float>(i, 2);
    float tx = preds.at<float>(i, 0), ty = preds.at<float>(i, 1);
    float tw = preds.at<float>(i, 2), th = preds.at<float>(i, 3);
    float cx = (tx + gx) * s, cy = (ty + gy) * s;
    float bw = tw * s, bh = th * s;

    // letterbox 像素 -> 原图像素
    float x1 = (cx - bw / 2 - dx) / r;
    float y1 = (cy - bh / 2 - dy) / r;
    float w = bw / r, h = bh / r;

    boxes.push_back({cvRound(x1), cvRound(y1), cvRound(w), cvRound(h)});
    scores.push_back(score);
    class_ids.push_back(class_id);
  }

  if (boxes.empty()) return detections;

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, keep);
  for (int i : keep) detections.push_back({boxes[i], scores[i]});

  return detections;
}

}  // namespace nailong
