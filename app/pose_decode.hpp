#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include "xtensor/xarray.hpp"

namespace pose {

namespace decode {

std::pair<float, xt::xarray<float>>
traverse_to_target_keypoint(size_t edge_id, xt::xarray<float> source_keypoints,
                            size_t target_keypoint_id, xt::xarray<float> scores,
                            xt::xarray<size_t> offsets, size_t output_stride,
                            xt::xarray<size_t> displacements);

std::pair<xt::xarray<float>, xt::xarray<size_t>> decode_pose(
    float root_score, size_t root_id, xt::xarray<size_t> root_image_coord,
    xt::xarray<float> scores, xt::xarray<size_t> offsets, size_t output_stride,
    xt::xarray<size_t> displacements_fwd, xt::xarray<size_t> displacements_bwd);

} // namespace decode

namespace decode_multi {

bool within_nms_radius_fast(xt::xarray<float> pose_coords,
                            size_t squared_nms_radius, xt::xarray<float> point);

float get_instance_score_fast(xt::xarray<float> exist_pose_coords,
                              size_t squared_nms_radius,
                              xt::xarray<float> keypoint_scores,
                              xt::xarray<float> keypoint_coords);

bool score_is_max_in_local_window(size_t keypoint_id, float score, size_t hmy,
                                  size_t hmx, size_t local_max_radius,
                                  xt::xarray<float> scores);

class Part {
public:
  explicit Part(float score, size_t keypoint_id, size_t y, size_t x);

  float score() const;
  size_t keypoint_id() const;
  size_t y() const;
  size_t x() const;

private:
  float score_;
  size_t keypoint_id_;
  size_t y_;
  size_t x_;
};

std::vector<Part> build_part_with_score(float score_threshold,
                                        size_t local_max_radius,
                                        xt::xarray<float> scores);

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>>
decode_multiple_poses(xt::xarray<float> scores, xt::xarray<float> offsets,
                      xt::xarray<float> displacements_fwd,
                      xt::xarray<float> displacements_bwd, size_t output_stride,
                      size_t max_pose_detections, float score_threshold,
                      size_t nms_radius, float min_pose_score);
} // namespace decode_multi
} // namespace pose
