#include "edgetpu.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/posenet/posenet_decoder_op.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor/xaxis_slice_iterator.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xview.hpp"

#include <boost/algorithm/string/join.hpp>
#include <boost/format.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

namespace po = boost::program_options;

template <typename T>
using Milliseconds =
    std::chrono::duration<T, std::chrono::milliseconds::period>;

template <typename T>
using Seconds = std::chrono::duration<T, std::chrono::seconds::period>;

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

class Keypoint {
public:
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
    RightAnkle,
    NumKeypoints
  };

public:
  explicit Keypoint() {}
  explicit Keypoint(Kind keypoint, cv::Point2f point, float score)
      : keypoint_(keypoint), point_(point), score_(score) {}

  Kind kind() const { return keypoint_; }
  cv::Point2f &point() { return point_; }
  cv::Point2f point() const { return point_; }
  float score() const { return score_; }

private:
  Kind keypoint_;
  cv::Point2f point_;
  float score_;
};

static constexpr auto NUM_KEYPOINTS =
    static_cast<std::size_t>(Keypoint::Kind::NumKeypoints);

using Keypoints = std::array<Keypoint, NUM_KEYPOINTS>;

class Pose {
public:
  explicit Pose(Keypoints keypoints, float score)
      : keypoints_(keypoints), score_(score) {}

  Keypoints keypoints() const { return keypoints_; }
  float score() const { return score_; }

private:
  Keypoints keypoints_;
  float score_;
};

using Poses = std::vector<Pose>;

class Engine {
public:
  explicit Engine(std::string model_path, bool mirror = false)
      : engine_(model_path),
        input_tensor_shape_(engine_.get_input_tensor_shape()),
        output_offsets_(Engine::make_output_offsets(
            engine_.get_all_output_tensors_sizes())),
        mirror_(mirror) {
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

  std::size_t width() const { return input_tensor_shape_[2]; }

  std::size_t depth() const { return input_tensor_shape_[3]; }

  std::pair<Poses, Milliseconds<float>> detect_poses(const cv::Mat &raw_img) {
    const std::size_t nbytes = raw_img.step[0] * raw_img.rows;
    std::vector<uint8_t> input(raw_img.data, raw_img.data + nbytes);
    auto outputs = engine_.RunInference(input);

    Milliseconds<float> inf_time(engine_.get_inference_time());

    auto keypoints = xt::adapt(
        outputs[0],
        std::vector<std::size_t>{
            {outputs[0].size() / NUM_KEYPOINTS / 2, NUM_KEYPOINTS, 2}});

    auto keypoint_scores = xt::adapt(
        outputs[1], std::vector<std::size_t>{
                        {outputs[1].size() / NUM_KEYPOINTS, NUM_KEYPOINTS}});
    auto pose_scores = outputs[2];
    auto nposes = static_cast<std::size_t>(outputs[3][0]);

    Poses poses;
    poses.reserve(nposes);

    for (auto [keypoint_ptr, pose_i] =
             std::make_pair(xt::axis_begin(keypoints, 0), 0);
         keypoint_ptr != xt::axis_end(keypoints, 0); ++keypoint_ptr, ++pose_i) {
      Keypoints keypoint_map;

      for (auto [point_ptr, point_i] =
               std::make_pair(xt::axis_slice_begin(*keypoint_ptr, 1), 0);
           point_ptr != xt::axis_slice_end(*keypoint_ptr, 1);
           ++point_ptr, ++point_i) {
        Keypoint keypoint(static_cast<Keypoint::Kind>(point_i),
                          cv::Point2f((*point_ptr)[1], (*point_ptr)[0]),
                          keypoint_scores(pose_i, point_i));
        if (mirror_) {
          keypoint.point().x = width() - keypoint.point().x;
        }

        keypoint_map[point_i] = keypoint;
      }
      poses.emplace_back(keypoint_map, pose_scores[pose_i]);
    }

    return std::make_pair(poses, inf_time);
  }

private:
  static std::vector<std::size_t>
  make_output_offsets(const std::vector<std::size_t> &output_sizes) {
    std::vector<std::size_t> result(output_sizes.size() + 1, 0);
    std::partial_sum(output_sizes.cbegin(), output_sizes.cend(),
                     result.begin() + 1);
    return result;
  }

private:
  coral::BasicEngine engine_;
  const std::vector<int32_t> input_tensor_shape_;
  const std::vector<std::size_t> output_offsets_;
  bool mirror_;
}; // namespace pose

namespace {
static constexpr std::size_t NUM_EDGES = 19;
using KeypointEdge = std::pair<Keypoint::Kind, Keypoint::Kind>;
} // namespace

static constexpr std::array<KeypointEdge, NUM_EDGES> EDGES = {
    std::make_pair(Keypoint::Kind::Nose, Keypoint::Kind::LeftEye),
    std::make_pair(Keypoint::Kind::Nose, Keypoint::Kind::RightEye),
    std::make_pair(Keypoint::Kind::Nose, Keypoint::Kind::LeftEar),
    std::make_pair(Keypoint::Kind::Nose, Keypoint::Kind::RightEar),
    std::make_pair(Keypoint::Kind::LeftEar, Keypoint::Kind::LeftEye),
    std::make_pair(Keypoint::Kind::RightEar, Keypoint::Kind::RightEye),
    std::make_pair(Keypoint::Kind::LeftEye, Keypoint::Kind::RightEye),
    std::make_pair(Keypoint::Kind::LeftShoulder, Keypoint::Kind::RightShoulder),
    std::make_pair(Keypoint::Kind::LeftShoulder, Keypoint::Kind::LeftElbow),
    std::make_pair(Keypoint::Kind::LeftShoulder, Keypoint::Kind::LeftHip),
    std::make_pair(Keypoint::Kind::RightShoulder, Keypoint::Kind::RightElbow),
    std::make_pair(Keypoint::Kind::RightShoulder, Keypoint::Kind::RightHip),
    std::make_pair(Keypoint::Kind::LeftElbow, Keypoint::Kind::LeftWrist),
    std::make_pair(Keypoint::Kind::RightElbow, Keypoint::Kind::RightWrist),
    std::make_pair(Keypoint::Kind::LeftHip, Keypoint::Kind::RightHip),
    std::make_pair(Keypoint::Kind::LeftHip, Keypoint::Kind::LeftKnee),
    std::make_pair(Keypoint::Kind::RightHip, Keypoint::Kind::RightKnee),
    std::make_pair(Keypoint::Kind::LeftKnee, Keypoint::Kind::LeftAnkle),
    std::make_pair(Keypoint::Kind::RightKnee, Keypoint::Kind::RightAnkle),
};
} // namespace pose

int main(int argc, const char *argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Test a model with coral::BasicEngine")(
      "model-path,m", po::value<std::string>(),
      "path to a Tensorflow Lite model")("device-path,d",
                                         po::value<std::string>(),
                                         "path to a v4l2 compatible device")(
      "threshold,t", po::value<float>()->default_value(0.2f),
      "pose keypoint score threshold")("width,w", po::value<int32_t>())(
      "height,H", po::value<int32_t>());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  auto model_path = vm["model-path"].as<std::string>();
  auto threshold = vm["threshold"].as<float>();
  auto width = vm["width"].as<int32_t>();
  auto height = vm["height"].as<int32_t>();

  cv::VideoCapture capture(vm["device-path"].as<std::string>(), cv::CAP_V4L2);
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280.0);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720.0);

  cv::Mat in_frame, out_frame, stream_out_frame;

  cv::Size frame_dims{{width, height}};

  Seconds<float> inf_secs(0.0f);
  Seconds<float> cam_secs(0.0f);
  Seconds<float> resize_secs(0.0f);
  size_t nframes = 0;

  pose::Engine engine(model_path);

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

    auto [poses, inf_millis] = engine.detect_poses(out_frame);
    inf_secs += inf_millis;
    auto true_secs = cam_secs + resize_secs + inf_secs;

    for (auto pose : poses) {
      std::unordered_map<pose::Keypoint::Kind, cv::Point2f> xys;

      for (const auto &keypoint : pose.keypoints()) {
        if (keypoint.score() >= threshold) {
          xys[keypoint.kind()] = keypoint.point();
          cv::circle(out_frame, keypoint.point(), 6, cv::Scalar(0, 255, 0), -1);
        }
      }

      for (auto &[a, b] : pose::EDGES) {
        auto apair = xys.find(a);
        auto bpair = xys.find(b);
        if (apair != xys.end() && bpair != xys.end()) {
          auto axy = apair->second;
          auto bxy = bpair->second;
          cv::line(out_frame, axy, bxy, cv::Scalar(0, 255, 255), 2);
        }
      }
    }

    auto model_fps_text =
        boost::format("Model FPS: %.1f") % (nframes / inf_secs.count());
    cv::putText(out_frame, model_fps_text.str(), cv::Point(width / 2, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    auto cam_fps_text =
        boost::format("Cam FPS: %.1f") % (nframes / cam_secs.count());
    cv::putText(out_frame, cam_fps_text.str(), cv::Point(width / 2, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    auto resize_fps_text =
        boost::format("Resize FPS: %.1f") % (nframes / resize_secs.count());
    cv::putText(out_frame, resize_fps_text.str(), cv::Point(width / 2, 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    auto true_fps_text =
        boost::format("True FPS: %.1f") % (nframes / true_secs.count());
    cv::putText(out_frame, true_fps_text.str(), cv::Point(width / 2, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    cv::resize(out_frame, stream_out_frame, in_frame.size());
  }

  return 0;
}
