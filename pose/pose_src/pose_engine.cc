#include "src/cpp/basic/basic_engine.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor/xaxis_slice_iterator.hpp"
#include "xtensor/xview.hpp"

#include "pose_decode.hpp"
#include "pose_engine.hpp"
#include "pose_types.hpp"

using namespace xt::placeholders;

namespace {

constexpr uint8_t OUTPUT_STRIDE = 16;

std::tuple<xt::xtensor<float, 3>, xt::xtensor<float, 3>, xt::xtensor<float, 4>,
           xt::xtensor<float, 4>>
extract_outputs(size_t img_h, size_t img_w,
                std::vector<std::vector<float>> outputs) {
  auto height = 1 + (img_h - 1) / OUTPUT_STRIDE;
  auto width = 1 + (img_w - 1) / OUTPUT_STRIDE;
  auto heatmaps = xt::adapt(
      outputs[0], std::vector{height, width, pose::constants::NUM_KEYPOINTS});
  auto offsets = xt::adapt(outputs[1],
                           {height, width, pose::constants::NUM_KEYPOINTS * 2});
  auto raw_dsp = xt::adapt(outputs[2], {height, width, 4UL, 16UL});

  auto fwd =
      xt::view(raw_dsp, xt::all(), xt::all(), xt::range(_, 2), xt::all());
  auto bwd =
      xt::view(raw_dsp, xt::all(), xt::all(), xt::range(2, 4), xt::all());
  return std::make_tuple(1.0 / (1.0 + xt::exp(-heatmaps)), offsets, fwd, bwd);
}
} // namespace

namespace pose {

Engine::Engine(const std::string &model_path, bool mirror)
    : engine_(std::make_unique<coral::BasicEngine>(model_path)),
      input_tensor_shape_(engine_->get_input_tensor_shape()), mirror_(mirror) {
  if (input_tensor_shape_.size() != 4) {
    throw InputDimensionsError(input_tensor_shape_.size());
  }
  if (input_tensor_shape_[0] != 1) {
    throw FirstDimensionSizeError(input_tensor_shape_[0]);
  }
  if (depth() != 3) {
    throw DepthSizeError(depth());
  }
}

size_t Engine::height() const { return input_tensor_shape_[1]; }
size_t Engine::width() const { return input_tensor_shape_[2]; }
size_t Engine::depth() const { return input_tensor_shape_[3]; }

std::pair<Poses, Milliseconds<float>>
Engine::detect_poses(const std::vector<uint8_t> &input, size_t max_poses,
                     float score_threshold, size_t nms_radius,
                     float min_pose_score) {
  auto [heatmaps, offsets, displacements_fwd, displacements_bwd] =
      extract_outputs(height(), width(), engine_->RunInference(input));
  Milliseconds<float> inf_time(engine_->get_inference_time());

  auto [pose_scores, keypoint_scores, keypoints] =
      pose::decode::decode_multiple_poses(heatmaps, offsets, displacements_fwd,
                                          displacements_bwd, OUTPUT_STRIDE,
                                          max_poses, score_threshold,
                                          nms_radius, min_pose_score);

  const auto [nposes, nkeypoints, _] = keypoints.shape();

  Poses poses;
  poses.reserve(nposes);

  for (size_t pose_i = 0; pose_i < nposes; ++pose_i) {
    Keypoints keypoint_map;

    const auto pose = xt::view(keypoints, pose_i, xt::all(), xt::keep(1, 0));

    for (size_t point_i = 0; point_i < nkeypoints; ++point_i) {
      const auto point = xt::row(pose, point_i);
      Keypoint keypoint{pose::constants::PART_NAMES[point_i],
                        Point{point[0], point[1]},
                        keypoint_scores(pose_i, point_i)};
      if (mirror_) {
        keypoint.point().x = width() - keypoint.point().x;
      }

      keypoint_map[point_i] = keypoint;
    }
    poses.emplace_back(keypoint_map, pose_scores[pose_i]);
  }

  return std::make_pair(poses, inf_time);
}

} // namespace pose
