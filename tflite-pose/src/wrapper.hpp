#pragma once

#include "edgetpu.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/posenet/posenet_decoder_op.h"

#include <string>
#include <tuple>
#include <vector>
#include <cstdint>

#include "opencv2/core.hpp"

namespace pose {

class Engine {
public:
  explicit Engine(std::string model_path) : engine_(model_path) { }

  int32_t run_inference(std::vector<uint8_t> raw_img) {
    // auto outputs = engine_.RunInference(raw_img);
    // keypoints = outputs[0];
    // keypoint_scores = outputs[1];
    // pose_scores = outputs[2];
    return 1;
    // return static_cast<std::size_t>(outputs[3][0]);
  }

  float get_inference_time() const {
    return engine_.get_inference_time();
  }

  std::vector<int32_t> get_input_tensor_shape() const {
    return engine_.get_input_tensor_shape();
  }

  std::vector<std::size_t> get_all_output_tensors_sizes() const {
    return engine_.get_all_output_tensors_sizes();
  }

private:
  coral::BasicEngine engine_;
}; // namespace pose
}
