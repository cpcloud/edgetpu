#pragma once

#include <array>

namespace utils {
namespace detail {

template <class T, std::size_t N, std::size_t... I>
constexpr std::array<std::remove_cv_t<T>, N>
to_array_impl(T(&&a)[N], std::index_sequence<I...>) {
  return {{std::move(a[I])...}};
}

template <class T, std::size_t N, std::size_t... I>
constexpr std::array<std::pair<std::remove_cv_t<T>, std::size_t>, N>
to_array_pair_impl(const std::array<T, N> &a, std::index_sequence<I...>) {
  return {{std::make_pair(a[I], static_cast<size_t>(a[I]))...}};
}

template <class T, class U, std::size_t N, std::size_t... I>
constexpr std::array<std::pair<std::size_t, std::size_t>, N>
to_array_pair_as_indices_impl(const std::array<std::pair<T, U>, N> &a,
                              std::index_sequence<I...>) {
  return {
      {std::move(std::make_pair(static_cast<std::size_t>(a[I].first),
                                static_cast<std::size_t>(a[I].second)))...}};
}

} // namespace detail

template <class T, std::size_t N>
constexpr std::array<std::remove_cv_t<T>, N> to_array(T(&&a)[N]) {
  return detail::to_array_impl(std::move(a), std::make_index_sequence<N>{});
}

template <class T, std::size_t N>
constexpr std::array<std::pair<std::remove_cv_t<T>, std::size_t>, N>
to_array_pair(const std::array<T, N> &a) {
  return detail::to_array_pair_impl(a, std::make_index_sequence<N>{});
}

template <class T, class U, std::size_t N>
constexpr std::array<std::pair<std::size_t, std::size_t>, N>
to_array_pair_as_indices(const std::array<std::pair<T, U>, N> &a) {
  return detail::to_array_pair_as_indices_impl(a,
                                               std::make_index_sequence<N>{});
}
} // namespace utils

namespace pose {
namespace constants {
namespace keypoint {

enum class Kind : uint8_t {
  Nose,
  LeftEye,
  RightEye,
  LeftEar,
  RightEar,
  LeftShoulder,
  RightShoulder,
  LeftElbow,
  RightElbow,
  LeftWrist,
  RightWrist,
  LeftHip,
  RightHip,
  LeftKnee,
  RightKnee,
  LeftAnkle,
  RightAnkle
};
}

constexpr auto PART_NAMES = utils::to_array(
    {keypoint::Kind::Nose, keypoint::Kind::LeftEye, keypoint::Kind::RightEye,
     keypoint::Kind::LeftEar, keypoint::Kind::RightEar,
     keypoint::Kind::LeftShoulder, keypoint::Kind::RightShoulder,
     keypoint::Kind::LeftElbow, keypoint::Kind::RightElbow,
     keypoint::Kind::LeftWrist, keypoint::Kind::RightWrist,
     keypoint::Kind::LeftHip, keypoint::Kind::RightHip,
     keypoint::Kind::LeftKnee, keypoint::Kind::RightKnee,
     keypoint::Kind::LeftAnkle, keypoint::Kind::RightAnkle});

constexpr auto PART_IDS = utils::to_array_pair(PART_NAMES);

constexpr auto NUM_KEYPOINTS = PART_NAMES.size();

constexpr auto CONNECTED_PART_NAMES = utils::to_array(
    {std::make_pair(keypoint::Kind::LeftHip, keypoint::Kind::LeftShoulder),
     std::make_pair(keypoint::Kind::LeftElbow, keypoint::Kind::LeftShoulder),
     std::make_pair(keypoint::Kind::LeftElbow, keypoint::Kind::LeftWrist),
     std::make_pair(keypoint::Kind::LeftHip, keypoint::Kind::LeftKnee),
     std::make_pair(keypoint::Kind::LeftKnee, keypoint::Kind::LeftAnkle),
     std::make_pair(keypoint::Kind::RightHip, keypoint::Kind::RightShoulder),
     std::make_pair(keypoint::Kind::RightElbow, keypoint::Kind::RightShoulder),
     std::make_pair(keypoint::Kind::RightElbow, keypoint::Kind::RightWrist),
     std::make_pair(keypoint::Kind::RightHip, keypoint::Kind::RightKnee),
     std::make_pair(keypoint::Kind::RightKnee, keypoint::Kind::RightAnkle),
     std::make_pair(keypoint::Kind::LeftShoulder,
                    keypoint::Kind::RightShoulder),
     std::make_pair(keypoint::Kind::LeftHip, keypoint::Kind::RightHip)});

constexpr auto CONNECTED_PART_INDICES = utils::to_array_pair_as_indices(
    CONNECTED_PART_NAMES); // cast elements to size_t

constexpr auto LOCAL_MAXIMUM_RADIUS = 1;
constexpr auto POSE_CHAIN = utils::to_array(
    {std::make_pair(keypoint::Kind::Nose, keypoint::Kind::LeftEye),
     std::make_pair(keypoint::Kind::LeftEye, keypoint::Kind::LeftEar),
     std::make_pair(keypoint::Kind::Nose, keypoint::Kind::RightEye),
     std::make_pair(keypoint::Kind::RightEye, keypoint::Kind::RightEar),
     std::make_pair(keypoint::Kind::Nose, keypoint::Kind::LeftShoulder),
     std::make_pair(keypoint::Kind::LeftShoulder, keypoint::Kind::LeftElbow),
     std::make_pair(keypoint::Kind::LeftElbow, keypoint::Kind::LeftWrist),
     std::make_pair(keypoint::Kind::LeftShoulder, keypoint::Kind::LeftHip),
     std::make_pair(keypoint::Kind::LeftHip, keypoint::Kind::LeftKnee),
     std::make_pair(keypoint::Kind::LeftKnee, keypoint::Kind::LeftAnkle),
     std::make_pair(keypoint::Kind::Nose, keypoint::Kind::RightShoulder),
     std::make_pair(keypoint::Kind::RightShoulder, keypoint::Kind::RightElbow),
     std::make_pair(keypoint::Kind::RightElbow, keypoint::Kind::RightWrist),
     std::make_pair(keypoint::Kind::RightShoulder, keypoint::Kind::RightHip),
     std::make_pair(keypoint::Kind::RightHip, keypoint::Kind::RightKnee),
     std::make_pair(keypoint::Kind::RightKnee, keypoint::Kind::RightAnkle)});

constexpr auto PARENT_CHILD_TUPLES =
    utils::to_array_pair_as_indices(POSE_CHAIN);
constexpr auto NUM_EDGES = PARENT_CHILD_TUPLES.size();

} // namespace constants
} // namespace pose
