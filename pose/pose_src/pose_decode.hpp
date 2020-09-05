#pragma once

#include "xtensor/xtensor.hpp"

namespace pose {
namespace decode {

using PoseScores = xt::xtensor<float, 1>;
using PoseKeypointScores = xt::xtensor<float, 2>;
using PoseKeypointCoords = xt::xtensor<float, 3>;

std::tuple<PoseScores, PoseKeypointScores, PoseKeypointCoords>
decode_multiple_poses(xt::xtensor<float, 3> scores,
                      xt::xtensor<float, 3> offsets,
                      xt::xtensor<float, 4> displacements_fwd,
                      xt::xtensor<float, 4> displacements_bwd,
                      size_t output_stride, size_t max_pose_detections,
                      float score_threshold, size_t nms_radius,
                      float min_pose_score);
} // namespace decode
} // namespace pose
