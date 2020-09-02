#include "pose_types.hpp"
#include "pose_constants.hpp"

namespace pose {

InputDimensionsError::InputDimensionsError(size_t ndims)
    : std::exception(), ndims_(ndims) {}

FirstDimensionSizeError::FirstDimensionSizeError(size_t first_dim_numel)
    : std::exception(), first_dim_numel_(first_dim_numel) {}

DepthSizeError::DepthSizeError(size_t depth)
    : std::exception(), depth_(depth) {}

Keypoint::Keypoint(keypoint::Kind kind, Point point, float score)
    : kind_(kind), point_(point), score_(score) {}

keypoint::Kind Keypoint::kind() const { return kind_; }
size_t Keypoint::index() const { return static_cast<size_t>(kind_); }
Point &Keypoint::point() { return point_; }
Point Keypoint::point() const { return point_; }
float Keypoint::score() const { return score_; }

Pose::Pose(Keypoints keypoints, float score)
    : keypoints_(keypoints), score_(score) {}
Keypoints Pose::keypoints() const { return keypoints_; }
float Pose::score() const { return score_; }

} // namespace pose
