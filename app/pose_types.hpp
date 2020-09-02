#pragma once

#include <array>
#include <chrono>
#include <vector>

#include "pose_constants.hpp"

namespace pose {

namespace keypoint = pose::constants::keypoint;

struct Point {
  float x;
  float y;
};

class InputDimensionsError : public std::exception {
public:
  explicit InputDimensionsError(size_t ndims);

private:
  size_t ndims_;
};

class FirstDimensionSizeError : public std::exception {
public:
  explicit FirstDimensionSizeError(size_t first_dim_numel);

private:
  size_t first_dim_numel_;
};

class DepthSizeError : public std::exception {
public:
  explicit DepthSizeError(size_t depth);

private:
  size_t depth_;
};

class Keypoint {
public:
  explicit Keypoint() = default;
  explicit Keypoint(keypoint::Kind kind, Point point, float score);

  keypoint::Kind kind() const;
  size_t index() const;
  Point &point();
  Point point() const;
  float score() const;

private:
  keypoint::Kind kind_;
  Point point_;
  float score_;
};

using Keypoints = std::array<Keypoint, pose::constants::NUM_KEYPOINTS>;

class Pose {
public:
  explicit Pose(Keypoints keypoints, float score);
  Keypoints keypoints() const;
  float score() const;

private:
  Keypoints keypoints_;
  float score_;
};

using Poses = std::vector<Pose>;

template <typename T>
using Milliseconds =
    std::chrono::duration<T, std::chrono::milliseconds::period>;

template <typename T>
using Seconds = std::chrono::duration<T, std::chrono::seconds::period>;

} // namespace pose
