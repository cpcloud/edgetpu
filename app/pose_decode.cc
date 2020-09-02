#include <functional>
#include <iostream>
#include <tuple>

#include "xtensor/xarray.hpp"
#include "xtensor/xindex_view.hpp"

#include "pose_constants.hpp"
#include "pose_decode.hpp"

using namespace xt::placeholders;

namespace pose {
namespace decode {

std::pair<float, xt::xarray<float>>
traverse_to_target_keypoint(size_t edge_id, xt::xarray<float> source_keypoints,
                            size_t target_keypoint_id, xt::xarray<float> scores,
                            xt::xarray<size_t> offsets, size_t output_stride,
                            xt::xarray<size_t> displacements) {
  auto height = scores.shape(0);
  auto width = scores.shape(1);

  auto rounded_source_keypoints = xt::round(source_keypoints / output_stride);
  auto source_keypoint_indices = xt::cast<size_t>(xt::xarray<float>{
      {xt::clip(xt::xscalar(rounded_source_keypoints[0]), 0.0f, height - 1.0f),
       xt::clip(xt::xscalar(rounded_source_keypoints[1]), 0.0f,
                width - 1.0f)}});
  auto displaced_points =
      source_keypoints + displacements(source_keypoint_indices[0],
                                       source_keypoint_indices[1], edge_id);
  auto displaced_point_indices = xt::cast<ssize_t>(xt::xarray<float>{
      {xt::clip(xt::xscalar(displaced_points[0]), 0.0f, height - 1.0f),
       xt::clip(xt::xscalar(displaced_points[1]), 0.0f, width - 1.0f)}});
  auto score = scores(displaced_point_indices[0], displaced_point_indices[1],
                      target_keypoint_id);
  auto image_coord = displaced_point_indices * output_stride *
                     offsets(displaced_point_indices[0],
                             displaced_point_indices[1], target_keypoint_id);
  return std::make_pair(score, image_coord);
}

std::pair<xt::xarray<float>, xt::xarray<size_t>>
decode_pose(float root_score, size_t root_id,
            xt::xarray<size_t> root_image_coord, xt::xarray<float> scores,
            xt::xarray<size_t> offsets, size_t output_stride,
            xt::xarray<size_t> displacements_fwd,
            xt::xarray<size_t> displacements_bwd) {
  auto num_parts = scores.shape(2);
  xt::xarray<float> instance_keypoint_scores = xt::zeros<float>({num_parts});
  xt::xarray<float> instance_keypoint_coords =
      xt::zeros<float>({num_parts, 2UL});
  instance_keypoint_scores[root_id] = root_score;
  xt::row(instance_keypoint_coords, root_id) = root_image_coord;

  constexpr auto EDGES = pose::constants::PARENT_CHILD_TUPLES;

  for (auto [edge, edge_iter] = std::make_pair(0, std::rbegin(EDGES));
       edge_iter != std::rend(EDGES); ++edge, ++edge_iter) {
    auto [target_keypoint_id, source_keypoint_id] = *edge_iter;
    if (instance_keypoint_scores[source_keypoint_id] > 0.0 &&
        instance_keypoint_scores[target_keypoint_id] == 0.0) {
      auto source_keypoints =
          xt::row(instance_keypoint_coords, source_keypoint_id);
      auto [score, coords] = traverse_to_target_keypoint(
          edge, source_keypoints, target_keypoint_id, scores, offsets,
          output_stride, displacements_bwd);
      instance_keypoint_scores[target_keypoint_id] = score;
      xt::view(instance_keypoint_coords, target_keypoint_id, 0) = coords[0];
      xt::view(instance_keypoint_coords, target_keypoint_id, 1) = coords[1];
    }
  }

  for (auto [edge, edge_iter] = std::make_pair(0, std::begin(EDGES));
       edge_iter != std::end(EDGES); ++edge, ++edge_iter) {
    auto [source_keypoint_id, target_keypoint_id] = *edge_iter;
    if (instance_keypoint_scores[source_keypoint_id] > 0.0 &&
        instance_keypoint_scores[target_keypoint_id] == 0.0) {
      auto source_keypoints =
          xt::row(instance_keypoint_coords, source_keypoint_id);
      auto [score, coords] = traverse_to_target_keypoint(
          edge, source_keypoints, target_keypoint_id, scores, offsets,
          output_stride, displacements_fwd);
      instance_keypoint_scores[target_keypoint_id] = score;
      xt::view(instance_keypoint_coords, target_keypoint_id, 0) = coords[0];
      xt::view(instance_keypoint_coords, target_keypoint_id, 1) = coords[1];
    }
  }

  return std::make_pair(instance_keypoint_scores, instance_keypoint_coords);
}
} // namespace decode
namespace decode_multi {

bool within_nms_radius_fast(xt::xarray<float> pose_coords,
                            size_t squared_nms_radius,
                            xt::xarray<float> point) {
  if (pose_coords.shape(0) == 0) {
    return false;
  }
  return xt::any(xt::sum(xt::pow(pose_coords - point, 2), 1) <=
                 squared_nms_radius);
}

float get_instance_score_fast(xt::xarray<float> exist_pose_coords,
                              size_t squared_nms_radius,
                              xt::xarray<float> keypoint_scores,
                              xt::xarray<float> keypoint_coords) {
  xt::xarray<float> scores;
  if (exist_pose_coords.shape(0) != 0) {
    scores = xt::filter(
        keypoint_scores,
        xt::reduce(std::logical_and{},
                   xt::sum(xt::pow(exist_pose_coords - keypoint_coords, 2), 2) >
                       squared_nms_radius,
                   {0}));
  } else {
    scores = keypoint_scores;
  }
  return xt::sum(scores)() / keypoint_scores.size();
}

bool score_is_max_in_local_window(size_t keypoint_id, float score, size_t hmy,
                                  size_t hmx, size_t local_max_radius,
                                  xt::xarray<float> scores) {
  auto height = scores.shape(0);
  auto width = scores.shape(1);
  auto y_start = std::max(hmy - local_max_radius, 0UL);
  auto y_end = std::min(hmy + local_max_radius + 1UL, height);
  auto x_start = std::max(hmx - local_max_radius, 0UL);
  auto x_end = std::min(hmx + local_max_radius + 1UL, width);

  for (size_t y = y_start; y < y_end; ++y) {
    for (size_t x = x_start; x < x_end; ++x) {
      if (scores(y, x, keypoint_id) > score) {
        return false;
      }
    }
  }

  return true;
}

Part::Part(float score, size_t keypoint_id, size_t y, size_t x)
    : score_(score), keypoint_id_(keypoint_id), y_(y), x_(x) {}

float Part::score() const { return score_; }
size_t Part::keypoint_id() const { return keypoint_id_; }
size_t Part::y() const { return y_; }
size_t Part::x() const { return x_; }

std::vector<Part> build_part_with_score(float score_threshold,
                                        size_t local_max_radius,
                                        xt::xarray<float> scores) {
  std::vector<Part> parts;
  auto height = scores.shape(0);
  auto width = scores.shape(1);
  auto num_keypoints = scores.shape(2);

  for (size_t hmy = 0; hmy < height; ++hmy) {
    for (size_t hmx = 0; hmx < width; ++hmx) {
      for (size_t keypoint_id = 0; keypoint_id < num_keypoints; ++keypoint_id) {
        auto score = scores(hmy, hmx, keypoint_id);
        if (score >= score_threshold &&
            score_is_max_in_local_window(keypoint_id, score, hmy, hmx,
                                         local_max_radius, scores)) {
          parts.emplace_back(score, keypoint_id, hmy, hmx);
        }
      }
    }
  }
  return parts;
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>>
decode_multiple_poses(xt::xarray<float> scores, xt::xarray<float> offsets,
                      xt::xarray<float> displacements_fwd,
                      xt::xarray<float> displacements_bwd, size_t output_stride,
                      size_t max_pose_detections, float score_threshold,
                      size_t nms_radius, float min_pose_score) {
  xt::xarray<float> pose_scores = xt::zeros<float>({max_pose_detections});
  xt::xarray<float> pose_keypoint_scores =
      xt::zeros<float>({max_pose_detections, pose::constants::NUM_KEYPOINTS});
  xt::xarray<float> pose_keypoint_coords = xt::zeros<float>(
      {max_pose_detections, pose::constants::NUM_KEYPOINTS, 2UL});
  const auto squared_nms_radius = nms_radius * nms_radius;
  auto scored_parts = build_part_with_score(
      score_threshold, pose::constants::LOCAL_MAXIMUM_RADIUS, scores);

  std::sort(std::begin(scored_parts), std::end(scored_parts),
            [](auto a, auto b) { return a.score() > b.score(); });

  const auto height = scores.shape(0);
  const auto width = scores.shape(1);
  std::vector new_shape{
      {static_cast<ssize_t>(height), static_cast<ssize_t>(width), 2L, -1L}};

  const size_t transpose_axes[] = {0, 1, 3, 2};

  auto new_offsets = xt::transpose(offsets.reshape(new_shape), transpose_axes);
  auto new_displacements_fwd =
      xt::transpose(displacements_fwd.reshape(new_shape), transpose_axes);
  auto new_displacements_bwd =
      xt::transpose(displacements_bwd.reshape(new_shape), transpose_axes);

  for (auto [pose_count, scored_part] =
           std::make_pair(0UL, std::begin(scored_parts));
       pose_count < max_pose_detections &&
       scored_part != std::end(scored_parts);
       ++pose_count, ++scored_part) {
    auto root_id = scored_part->keypoint_id();
    auto y = scored_part->y();
    auto x = scored_part->x();
    auto root_score = scored_part->score();
    auto root_image_coords =
        xt::xarray<size_t>{y, x} * output_stride * new_offsets(y, x, root_id);
    if (within_nms_radius_fast(
            xt::view(pose_keypoint_coords, xt::range(_, pose_count), root_id),
            squared_nms_radius, root_image_coords)) {
      continue;
    }
    auto [keypoint_scores, keypoint_coords] = decode::decode_pose(
        root_score, root_id, root_image_coords, scores, new_offsets,
        output_stride, new_displacements_fwd, new_displacements_bwd);
    auto pose_score = get_instance_score_fast(
        xt::view(pose_keypoint_coords, xt::range(_, pose_count)),
        squared_nms_radius, keypoint_scores, keypoint_coords);
    pose_scores[pose_count] = pose_score;

    xt::row(pose_keypoint_scores, pose_count) = keypoint_scores;
    xt::view(pose_keypoint_coords, pose_count) = keypoint_coords;
  }
  return std::make_tuple(pose_scores, pose_keypoint_scores,
                         pose_keypoint_coords);
}
} // namespace decode_multi
} // namespace pose
