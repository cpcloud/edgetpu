#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "src/cpp/basic/basic_engine.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor/xaxis_slice_iterator.hpp"

#include <boost/format.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

namespace po = boost::program_options;

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

constexpr auto NOSE_VARIANT = Keypoint::Kind(NOSE);
constexpr auto NOSE_INDEX = NOSE_VARIANT.index();

constexpr auto LEFT_EYE_VARIANT = Keypoint::Kind(LEFT_EYE);
constexpr auto LEFT_EYE_INDEX = LEFT_EYE_VARIANT.index();

constexpr auto RIGHT_EYE_INDEX = Keypoint::Kind(RIGHT_EYE).index();
constexpr auto LEFT_EAR_INDEX = Keypoint::Kind(LEFT_EAR).index();
constexpr auto RIGHT_EAR_INDEX = Keypoint::Kind(RIGHT_EAR).index();
constexpr auto LEFT_SHOULDER_INDEX = Keypoint::Kind(LEFT_SHOULDER).index();
constexpr auto RIGHT_SHOULDER_INDEX = Keypoint::Kind(RIGHT_SHOULDER).index();
constexpr auto LEFT_ELBOW_INDEX = Keypoint::Kind(LEFT_ELBOW).index();
constexpr auto RIGHT_ELBOW_INDEX = Keypoint::Kind(RIGHT_ELBOW).index();
constexpr auto LEFT_WRIST_INDEX = Keypoint::Kind(LEFT_WRIST).index();
constexpr auto RIGHT_WRIST_INDEX = Keypoint::Kind(RIGHT_WRIST).index();
constexpr auto LEFT_HIP_INDEX = Keypoint::Kind(LEFT_HIP).index();
constexpr auto RIGHT_HIP_INDEX = Keypoint::Kind(RIGHT_HIP).index();

static constexpr std::size_t NUM_KEYPOINTS =
    std::variant_size_v<Keypoint::Kind>;

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

Keypoint::Kind position_to_keypoint_kind(std::size_t i) {
  switch (i) {
  case NOSE_INDEX:
    return NOSE_VARIANT;
  case LEFT_EYE_INDEX:
    return LEFT_EYE_VARIANT;
  case RIGHT_EYE_INDEX:
    return RIGHT_EYE_VARIANT;
  case 3:
    return Keypoint::Kind(std::variant_alternative_t<RIGHT_EYE_INDEX, Keypoint::Kind>{});
  case 4:
    return Keypoint::Kind(std::variant_alternative_t<4, Keypoint::Kind>{});
  case 5:
    return Keypoint::Kind(std::variant_alternative_t<5, Keypoint::Kind>{});
  case 6:
    return Keypoint::Kind(std::variant_alternative_t<6, Keypoint::Kind>{});
  case 7:
    return Keypoint::Kind(std::variant_alternative_t<7, Keypoint::Kind>{});
  case 8:
    return Keypoint::Kind(std::variant_alternative_t<8, Keypoint::Kind>{});
  case 9:
    return Keypoint::Kind(std::variant_alternative_t<9, Keypoint::Kind>{});
  case 10:
    return Keypoint::Kind(std::variant_alternative_t<10, Keypoint::Kind>{});
  case 11:
    return Keypoint::Kind(std::variant_alternative_t<11, Keypoint::Kind>{});
  case 12:
    return Keypoint::Kind(std::variant_alternative_t<12, Keypoint::Kind>{});
  case 13:
    return Keypoint::Kind(std::variant_alternative_t<13, Keypoint::Kind>{});
  case 14:
    return Keypoint::Kind(std::variant_alternative_t<14, Keypoint::Kind>{});
  case 15:
    return Keypoint::Kind(std::variant_alternative_t<15, Keypoint::Kind>{});
  case 16:
    return Keypoint::Kind(std::variant_alternative_t<16, Keypoint::Kind>{});
  default:
    throw std::exception();
  }
}

class Engine {
public:
  explicit Engine(std::string model_path, bool mirror = false)
      : engine_(model_path),
        input_tensor_shape_(engine_.get_input_tensor_shape()), mirror_(mirror) {
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

  std::pair<std::vector<Pose>, Milliseconds>
  detect_poses(const cv::Mat &raw_img) {
    const std::size_t nbytes = raw_img.step[0] * raw_img.rows;
    auto outputs = engine_.RunInference(
        std::vector<uint8_t>(raw_img.data, raw_img.data + nbytes));

    Milliseconds inf_time(engine_.get_inference_time());

    auto keypoints = xt::adapt(
        outputs[0],
        std::vector<std::size_t>{
            {outputs[0].size() / NUM_KEYPOINTS / 2, NUM_KEYPOINTS, 2}});

    auto keypoint_scores = xt::adapt(
        outputs[1], std::vector<std::size_t>{
                        {outputs[1].size() / NUM_KEYPOINTS, NUM_KEYPOINTS}});
    auto pose_scores = outputs[2];
    auto nposes = static_cast<std::size_t>(outputs[3][0]);

    std::vector<Pose> poses;
    poses.reserve(nposes);

    for (auto [keypoint_ptr, pose_i] =
             std::make_pair(xt::axis_begin(keypoints, 0), 0);
         keypoint_ptr != xt::axis_end(keypoints, 0); ++keypoint_ptr, ++pose_i) {
      std::array<Keypoint, NUM_KEYPOINTS> keypoint_map;

      for (auto [point_ptr, point_i] =
               std::make_pair(xt::axis_slice_begin(*keypoint_ptr, 1), 0);
           point_ptr != xt::axis_slice_end(*keypoint_ptr, 1);
           ++point_ptr, ++point_i) {
        Keypoint keypoint(position_to_keypoint_kind(point_i),
                          cv::Point2f((*point_ptr)[1], (*point_ptr)[0]),
                          keypoint_scores(pose_i, point_i));
        if (mirror_) {
          keypoint.point.x = width() - keypoint.point.x;
        }

        keypoint_map[point_i] = keypoint;
      }
      poses.emplace_back(keypoint_map, pose_scores[pose_i]);
    }

    return std::make_pair(poses, inf_time);
  }

private:
  coral::BasicEngine engine_;
  const std::vector<int32_t> input_tensor_shape_;
  bool mirror_;
}; // namespace pose

static constexpr std::size_t NUM_EDGES = 19;
using KeypointEdge = std::pair<Keypoint::Kind, Keypoint::Kind>;

static constexpr std::array<KeypointEdge, NUM_EDGES> EDGES = {
    std::make_pair(Keypoint::Kind(NOSE), Keypoint::Kind(LeftEye())),
    std::make_pair(Keypoint::Kind(NOSE), Keypoint::Kind(RightEye())),
    std::make_pair(Keypoint::Kind(NOSE), Keypoint::Kind(LeftEar())),
    std::make_pair(Keypoint::Kind(NOSE), Keypoint::Kind(RightEar())),
    std::make_pair(Keypoint::Kind(LeftEar()), Keypoint::Kind(LeftEye())),
    std::make_pair(Keypoint::Kind(RightEar()), Keypoint::Kind(RightEye())),
    std::make_pair(Keypoint::Kind(LeftEye()), Keypoint::Kind(RightEye())),
    std::make_pair(Keypoint::Kind(LeftShoulder()),
                   Keypoint::Kind(RightShoulder())),
    std::make_pair(Keypoint::Kind(LeftShoulder()), Keypoint::Kind(LeftElbow())),
    std::make_pair(Keypoint::Kind(LeftShoulder()), Keypoint::Kind(LeftHip())),
    std::make_pair(Keypoint::Kind(RightShoulder()),
                   Keypoint::Kind(RightElbow())),
    std::make_pair(Keypoint::Kind(RightShoulder()), Keypoint::Kind(RightHip())),
    std::make_pair(Keypoint::Kind(LeftElbow()), Keypoint::Kind(LeftWrist())),
    std::make_pair(Keypoint::Kind(RightElbow()), Keypoint::Kind(RightWrist())),
    std::make_pair(Keypoint::Kind(LeftHip()), Keypoint::Kind(RightHip())),
    std::make_pair(Keypoint::Kind(LeftHip()), Keypoint::Kind(LeftKnee())),
    std::make_pair(Keypoint::Kind(RightHip()), Keypoint::Kind(RightKnee())),
    std::make_pair(Keypoint::Kind(LeftKnee()), Keypoint::Kind(LeftAnkle())),
    std::make_pair(Keypoint::Kind(RightKnee()), Keypoint::Kind(RightAnkle())),
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
      "pose keypoint score threshold")("width,w", po::value<int32_t>(),
                                       "output frame width")(
      "height,H", po::value<int32_t>(), "output frame height");
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

  Seconds inf_secs(0.0f);
  Seconds cam_secs(0.0f);
  Seconds resize_secs(0.0f);
  std::size_t nframes = 0;

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
