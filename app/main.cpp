#include <array>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"

#include "coral/tflite_utils.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor/xaxis_slice_iterator.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

using Microseconds =
    std::chrono::duration<float, std::chrono::microseconds::period>;
using Milliseconds =
    std::chrono::duration<float, std::chrono::milliseconds::period>;
using Seconds = std::chrono::duration<float, std::chrono::seconds::period>;

namespace pose {

class InputDimensionsError : public std::exception {
public:
  explicit InputDimensionsError(std::size_t ndims)
      : std::exception(), ndims_(ndims) {}

private:
  std::size_t ndims_;
};

class FirstDimensionSizeError : public std::exception {
public:
  explicit FirstDimensionSizeError(std::size_t first_dim_numel)
      : std::exception(), first_dim_numel_(first_dim_numel) {}

private:
  std::size_t first_dim_numel_;
};

class DepthSizeError : public std::exception {
public:
  explicit DepthSizeError(std::size_t depth)
      : std::exception(), depth_(depth) {}

private:
  std::size_t depth_;
};

struct Nose {};
constexpr auto NOSE = Nose{};

struct LeftEye {};
constexpr auto LEFT_EYE = LeftEye{};

struct RightEye {};
constexpr auto RIGHT_EYE = RightEye{};

struct LeftEar {};
constexpr auto LEFT_EAR = LeftEar{};

struct RightEar {};
constexpr auto RIGHT_EAR = RightEar{};

struct LeftShoulder {};
constexpr auto LEFT_SHOULDER = LeftShoulder{};

struct RightShoulder {};
constexpr auto RIGHT_SHOULDER = RightShoulder{};

struct LeftElbow {};
constexpr auto LEFT_ELBOW = LeftElbow{};

struct RightElbow {};
constexpr auto RIGHT_ELBOW = RightElbow{};

struct LeftWrist {};
constexpr auto LEFT_WRIST = LeftWrist{};

struct RightWrist {};
constexpr auto RIGHT_WRIST = RightWrist{};

struct LeftHip {};
constexpr auto LEFT_HIP = LeftHip{};

struct RightHip {};
constexpr auto RIGHT_HIP = RightHip{};

struct LeftKnee {};
constexpr auto LEFT_KNEE = LeftKnee{};

struct RightKnee {};
constexpr auto RIGHT_KNEE = RightKnee{};

struct LeftAnkle {};
constexpr auto LEFT_ANKLE = LeftAnkle{};

struct RightAnkle {};
constexpr auto RIGHT_ANKLE = RightAnkle{};

struct Keypoint {
  using Kind = std::variant<Nose, LeftEye, RightEye, LeftEar, RightEar,
                            LeftShoulder, RightShoulder, LeftElbow, RightElbow,
                            LeftWrist, RightWrist, LeftHip, RightHip, LeftKnee,
                            RightKnee, LeftAnkle, RightAnkle>;
  Kind kind;
  cv::Point2f point;
  float score;
};

#define DEFINE_KIND_CONSTS(kp)                                                 \
  constexpr auto kp##_VARIANT = Keypoint::Kind((kp));                          \
  constexpr auto kp##_INDEX = kp##_VARIANT.index()

DEFINE_KIND_CONSTS(NOSE);
DEFINE_KIND_CONSTS(LEFT_EYE);
DEFINE_KIND_CONSTS(RIGHT_EYE);
DEFINE_KIND_CONSTS(LEFT_EAR);
DEFINE_KIND_CONSTS(RIGHT_EAR);
DEFINE_KIND_CONSTS(LEFT_SHOULDER);
DEFINE_KIND_CONSTS(RIGHT_SHOULDER);
DEFINE_KIND_CONSTS(LEFT_ELBOW);
DEFINE_KIND_CONSTS(RIGHT_ELBOW);
DEFINE_KIND_CONSTS(LEFT_WRIST);
DEFINE_KIND_CONSTS(RIGHT_WRIST);
DEFINE_KIND_CONSTS(LEFT_HIP);
DEFINE_KIND_CONSTS(RIGHT_HIP);
DEFINE_KIND_CONSTS(LEFT_KNEE);
DEFINE_KIND_CONSTS(RIGHT_KNEE);
DEFINE_KIND_CONSTS(LEFT_ANKLE);
DEFINE_KIND_CONSTS(RIGHT_ANKLE);

#undef DEFINE_KIND_CONSTS

constexpr std::size_t NUM_KEYPOINTS = std::variant_size_v<Keypoint::Kind>;

struct Pose {
  std::array<Keypoint, NUM_KEYPOINTS> keypoints;
  float score;
};

std::vector<std::size_t>
make_output_offsets(std::vector<std::size_t> output_sizes) {
  std::vector<std::size_t> result(output_sizes.size() + 1, 0);
  std::partial_sum(output_sizes.cbegin(), output_sizes.cend(),
                   result.begin() + 1);
  return result;
}

#define CASE_POSITION_TO_KEYPOINT(kp)                                          \
  case kp##_INDEX:                                                             \
    return kp##_VARIANT

constexpr Keypoint::Kind position_to_keypoint_kind(std::size_t i) {
  switch (i) {
    CASE_POSITION_TO_KEYPOINT(NOSE);
    CASE_POSITION_TO_KEYPOINT(LEFT_EYE);
    CASE_POSITION_TO_KEYPOINT(RIGHT_EYE);
    CASE_POSITION_TO_KEYPOINT(LEFT_EAR);
    CASE_POSITION_TO_KEYPOINT(RIGHT_EAR);
    CASE_POSITION_TO_KEYPOINT(LEFT_SHOULDER);
    CASE_POSITION_TO_KEYPOINT(RIGHT_SHOULDER);
    CASE_POSITION_TO_KEYPOINT(LEFT_ELBOW);
    CASE_POSITION_TO_KEYPOINT(RIGHT_ELBOW);
    CASE_POSITION_TO_KEYPOINT(LEFT_WRIST);
    CASE_POSITION_TO_KEYPOINT(RIGHT_WRIST);
    CASE_POSITION_TO_KEYPOINT(LEFT_HIP);
    CASE_POSITION_TO_KEYPOINT(RIGHT_HIP);
    CASE_POSITION_TO_KEYPOINT(LEFT_KNEE);
    CASE_POSITION_TO_KEYPOINT(RIGHT_KNEE);
    CASE_POSITION_TO_KEYPOINT(LEFT_ANKLE);
    CASE_POSITION_TO_KEYPOINT(RIGHT_ANKLE);
  default:
    throw std::logic_error("invalid variant index");
  }
}

#undef CASE_POSITION_TO_KEYPOINT

class Engine {
public:
  explicit Engine(std::string model_path, bool mirror = false)
      : engine_(coral::MakeEdgeTpuInterpreterOrDie(
            *coral::LoadModelOrDie(model_path))),
        input_tensor_shape_(engine_->inputs()),
        output_tensor_shape_(engine_->outputs()), mirror_(mirror) {
    if (input_tensor_shape_.size() != 4) {
      throw InputDimensionsError(input_tensor_shape_.size());
    }
    if (input_tensor_shape_[0] != 1) {
      throw FirstDimensionSizeError(input_tensor_shape_[0]);
    }
    if (depth() != 3) {
      throw DepthSizeError(depth());
    }

    CHECK_EQ(engine_->AllocateTensors(), kTfLiteOk);
  }

  std::size_t width() const { return input_tensor_shape_[2]; }

  std::size_t depth() const { return input_tensor_shape_[3]; }

  std::tuple<std::vector<Pose>, Microseconds, Milliseconds, Microseconds>
  detect_poses(const cv::Mat &raw_img) {
    const std::size_t nbytes = raw_img.step[0] * raw_img.rows;

    const auto start_copy = std::chrono::steady_clock::now();
    std::copy(raw_img.data, raw_img.data + nbytes,
              engine_->typed_input_tensor<uint8_t>(0));
    const auto stop_copy = std::chrono::steady_clock::now();

    const auto start_inference = std::chrono::steady_clock::now();
    CHECK_EQ(engine_->Invoke(), kTfLiteOk);
    const auto stop_inference = std::chrono::steady_clock::now();

    const auto start_post_proc = std::chrono::steady_clock::now();
    auto pose_keypoints = xt::adapt(
        engine_->typed_output_tensor<float>(0),
        std::vector<std::size_t>{
            {output_tensor_shape_[0] / NUM_KEYPOINTS / 2, NUM_KEYPOINTS, 2}});

    auto keypoint_scores = xt::adapt(
        engine_->typed_output_tensor<float>(1),
        std::vector<std::size_t>{
            {output_tensor_shape_[1] / NUM_KEYPOINTS, NUM_KEYPOINTS}});
    auto pose_scores = engine_->typed_output_tensor<float>(2);
    auto nposes =
        static_cast<std::size_t>(engine_->typed_output_tensor<float>(3)[0]);

    std::vector<Pose> poses;
    poses.reserve(nposes);

    for (auto [keypoint_ptr, pose_i] =
             std::make_pair(xt::axis_begin(pose_keypoints, 0), 0);
         keypoint_ptr != xt::axis_end(pose_keypoints, 0);
         ++keypoint_ptr, ++pose_i) {
      std::array<Keypoint, NUM_KEYPOINTS> keypoint_map;

      for (auto [point_ptr, point_i] =
               std::make_pair(xt::axis_slice_begin(*keypoint_ptr, 1), 0);
           point_ptr != xt::axis_slice_end(*keypoint_ptr, 1);
           ++point_ptr, ++point_i) {
        Keypoint keypoint{position_to_keypoint_kind(point_i),
                          cv::Point2f((*point_ptr)[1], (*point_ptr)[0]),
                          keypoint_scores(pose_i, point_i)};
        if (mirror_) {
          keypoint.point.x = width() - keypoint.point.x;
        }

        keypoint_map[point_i] = keypoint;
      }
      poses.push_back(Pose{keypoint_map, pose_scores[pose_i]});
    }
    const auto stop_post_proc = std::chrono::steady_clock::now();

    return std::make_tuple(poses, stop_copy - start_copy,
                           stop_inference - start_inference,
                           stop_post_proc - start_post_proc);
  }

private:
  std::unique_ptr<tflite::Interpreter> engine_;
  const std::vector<int32_t> input_tensor_shape_;
  const std::vector<int32_t> output_tensor_shape_;
  bool mirror_;
}; // namespace pose

constexpr std::size_t NUM_EDGES = 19;
using KeypointEdge = std::pair<Keypoint::Kind, Keypoint::Kind>;

constexpr std::array<KeypointEdge, NUM_EDGES> EDGES = {
    std::make_pair(NOSE_VARIANT, LEFT_EYE_VARIANT),
    std::make_pair(NOSE_VARIANT, RIGHT_EYE_VARIANT),
    std::make_pair(NOSE_VARIANT, LEFT_EAR_VARIANT),
    std::make_pair(NOSE_VARIANT, RIGHT_EAR_VARIANT),
    std::make_pair(LEFT_EAR_VARIANT, LEFT_EYE_VARIANT),
    std::make_pair(RIGHT_EAR_VARIANT, RIGHT_EYE_VARIANT),
    std::make_pair(LEFT_EYE_VARIANT, RIGHT_EYE_VARIANT),
    std::make_pair(LEFT_SHOULDER_VARIANT, RIGHT_SHOULDER_VARIANT),
    std::make_pair(LEFT_SHOULDER_VARIANT, LEFT_ELBOW_VARIANT),
    std::make_pair(LEFT_SHOULDER_VARIANT, LEFT_HIP_VARIANT),
    std::make_pair(RIGHT_SHOULDER_VARIANT, RIGHT_ELBOW_VARIANT),
    std::make_pair(RIGHT_SHOULDER_VARIANT, RIGHT_HIP_VARIANT),
    std::make_pair(LEFT_ELBOW_VARIANT, LEFT_WRIST_VARIANT),
    std::make_pair(RIGHT_ELBOW_VARIANT, RIGHT_WRIST_VARIANT),
    std::make_pair(LEFT_HIP_VARIANT, RIGHT_HIP_VARIANT),
    std::make_pair(LEFT_HIP_VARIANT, LEFT_KNEE_VARIANT),
    std::make_pair(RIGHT_HIP_VARIANT, RIGHT_KNEE_VARIANT),
    std::make_pair(LEFT_KNEE_VARIANT, LEFT_ANKLE_VARIANT),
    std::make_pair(RIGHT_KNEE_VARIANT, RIGHT_ANKLE_VARIANT),
};

} // namespace pose

struct Path {
  std::string path;
};

std::string AbslUnparseFlag(Path p) { return absl::UnparseFlag(p.path); }

bool AbslParseFlag(absl::string_view text, Path *path,
                   std::string *error) {
  if (!absl::ParseFlag(text, &path->path, error)) {
    return false;
  }

  if (path->path.empty()) {
    *error = "provided path is empty";
    return false;
  }

  if (!std::filesystem::exists(std::filesystem::path(path->path))) {
    *error = "path provided does not exist";
    return false;
  }

  return true;
}

ABSL_FLAG(Path, model_path, Path{},
          "Path to a Tensorflow Lite model");
ABSL_FLAG(Path, device_path, Path{"/dev/video0"},
          "Path to a v4l2 supported device");
ABSL_FLAG(float, threshold, 0.2f, "Pose score threshold");
ABSL_FLAG(int32_t, width, -1.0, "Output frame width");
ABSL_FLAG(int32_t, height, -1.0, "Output frame height");

int main(int argc, char **argv) {
  absl::ParseCommandLine(argc, argv);

  auto model_path = absl::GetFlag(FLAGS_model_path);
  auto threshold = absl::GetFlag(FLAGS_threshold);
  auto width = absl::GetFlag(FLAGS_width);
  auto height = absl::GetFlag(FLAGS_height);
  auto device_path = absl::GetFlag(FLAGS_device_path);

  cv::VideoCapture capture(device_path.path, cv::CAP_V4L2);
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280.0);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720.0);

  cv::Mat in_frame, out_frame;

  cv::Size frame_dims{{width, height}};

  Seconds inf_secs(0.0f);
  Seconds cam_secs(0.0f);
  Seconds resize_secs(0.0f);
  std::size_t nframes = 0;

  pose::Engine engine(model_path.path);

  while (true) {
    auto start_frame = std::chrono::steady_clock::now();
    capture.read(in_frame);
    auto stop_frame = std::chrono::steady_clock::now();
    cam_secs += stop_frame - start_frame;

    ++nframes;

    auto start_resize = std::chrono::steady_clock::now();
    cv::resize(in_frame, out_frame, frame_dims, 0, 0,
               cv::InterpolationFlags::INTER_LINEAR);
    auto stop_resize = std::chrono::steady_clock::now();
    resize_secs += stop_resize - start_resize;

    auto [poses, copy_millis, inf_millis, post_proc_millis] =
        engine.detect_poses(out_frame);
    inf_secs += inf_millis;
    auto true_secs = cam_secs + resize_secs + inf_secs;

    for (auto pose : poses) {
      std::array<std::optional<cv::Point2f>, pose::NUM_KEYPOINTS> xys;

      for (const auto &keypoint : pose.keypoints) {
        if (keypoint.score >= threshold) {
          xys[keypoint.kind.index()] = keypoint.point;
          cv::circle(out_frame, keypoint.point, 6, cv::Scalar(0, 255, 0), -1);
        }
      }

      for (auto &[a, b] : pose::EDGES) {
        const auto a_point = xys[a.index()];
        const auto b_point = xys[b.index()];
        if (a_point.has_value() && b_point.has_value()) {
          cv::line(out_frame, a_point.value(), b_point.value(),
                   cv::Scalar(0, 255, 255), 2);
        }
      }
    }

    auto model_fps_text =
        absl::StrFormat("Model FPS: %.1f", nframes / inf_secs.count());
    cv::putText(out_frame, model_fps_text, cv::Point(width / 2, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    auto cam_fps_text =
        absl::StrFormat("Cam FPS: %.1f", nframes / cam_secs.count());
    cv::putText(out_frame, cam_fps_text, cv::Point(width / 2, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    auto resize_fps_text =
        absl::StrFormat("Resize FPS: %.1f", nframes / resize_secs.count());
    cv::putText(out_frame, resize_fps_text, cv::Point(width / 2, 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    auto true_fps_text =
        absl::StrFormat("True FPS: %.1f", nframes / true_secs.count());
    cv::putText(out_frame, true_fps_text, cv::Point(width / 2, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);
  }

  return 0;
}
