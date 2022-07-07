// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "Utils.h"                     // NOLINT
#include "paddle_api.h"                // NOLINT
#include <EGL/egl.h>                   // NOLINT
#include <GLES2/gl2.h>                 // NOLINT
#include <map>                         // NOLINT
#include <memory>                      // NOLINT
#include <opencv2/core.hpp>            // NOLINT
#include <opencv2/highgui/highgui.hpp> // NOLINT
#include <opencv2/imgcodecs.hpp>       // NOLINT
#include <opencv2/imgproc.hpp>         // NOLINT
#include <string>                      // NOLINT
#include <vector>                      // NOLINT

struct Object {
  std::string class_name;
  cv::Scalar fill_color;
  float prob;
  cv::Rect rec;
  int class_id;
};

class Detector {
public: // NOLINT
  explicit Detector(const std::string &modelDir, const std::string &labelPath,
                    const int cpuThreadNum, const std::string &cpuPowerMode,
                    int inputWidth, int inputHeight,
                    const std::vector<float> &inputMean,
                    const std::vector<float> &inputStd, float scoreThreshold);

  void Predict(const cv::Mat &rgbImage, std::vector<Object> *results,
               double *preprocessTime, double *predictTime,
               double *postprocessTime);

private: // NOLINT
  std::vector<std::string> LoadLabelList(const std::string &path);
  std::vector<cv::Scalar> GenerateColorMap(int numOfClasses);
  void Preprocess(const cv::Mat &rgbaImage);
  void Postprocess(std::vector<Object> *results);
  void Nms(const std::map<int, std::vector<Object>> &src,
           std::vector<Object> *res);

//  void ExtractBoxes(int seq_id, const float *in,
//                    std::map<int, std::vector<Object>> *outs,
//                    const std::vector<int64_t> &shape);

    void ExtractBoxes(const float *in,    // float[]
                                std::map<int, std::vector<Object>> *outs,
                                const std::vector<int64_t> &shape);

  void InitParams(const int &width, const int &height);

private: // NOLINT
  int inputWidth_;
  int inputHeight_;
  std::vector<float> inputMean_;
  std::vector<float> inputStd_;
  float scoreThreshold_;
  std::vector<cv::Scalar> colorMap_;
  std::vector<std::string> labelList_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;

  bool isInited_{false};
  int channelLength_;
  const int strides_[3] = {8, 16, 32};
  const int anchors_[3][6] = {{10, 13, 16, 30, 33, 23},
                              {30, 61, 62, 45, 59, 119},
                              {116, 90, 156, 198, 373, 326}};
  const float confThresh_ = 0.25;
  const float nmsThresh_ = 0.45;
  float ratio_ = 1.0;
  int inputX, inputY, inputW, inputH;
};

class Pipeline {
public: // NOLINT
  Pipeline(const std::string &modelDir, const std::string &labelPath,
           const int cpuThreadNum, const std::string &cpuPowerMode,
           int inputWidth, int inputHeight, const std::vector<float> &inputMean,
           const std::vector<float> &inputStd, float scoreThreshold);

  bool Process(cv::Mat &rgbaImage, std::string savedImagePath); // NOLINT

private: // NOLINT
  // Visualize the results to origin image
  void VisualizeResults(const std::vector<Object> &results, cv::Mat *rgbaImage);

  // Visualize the status(performace data) to origin image
  void VisualizeStatus(double preprocessTime, double predictTime,
                       double postprocessTime, cv::Mat *rgbaImage);

  void VisualizeResults_for_Eric(int a,int b,int c,int d,
                                 cv::Mat *rgbaImage);

private: // NOLINT
  std::shared_ptr<Detector> detector_;
};
