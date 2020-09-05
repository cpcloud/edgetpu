#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pose_types.hpp"

namespace coral {
class BasicEngine;
}

namespace pose {

class Engine {
public:
  explicit Engine(const std::string &model_path, bool mirror = false);

  size_t height() const;
  size_t width() const;
  size_t depth() const;

  std::pair<Poses, Milliseconds<float>>
  detect_poses(const std::vector<uint8_t> &input, size_t max_poses,
               float score_threshold, size_t nms_radius, float min_pose_score);

private:
  std::unique_ptr<coral::BasicEngine> engine_;
  const std::vector<int32_t> input_tensor_shape_;
  const bool mirror_;
};

} // namespace pose
