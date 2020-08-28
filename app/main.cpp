#include "edgetpu.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/posenet/posenet_decoder_op.h"

#include "xtensor-io/ximage.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xview.hpp"

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace po = boost::program_options;

namespace pose {

static constexpr std::array<const char *, 17> KEYPOINTS = {
    "nose",        "left eye",      "right eye",      "left ear",
    "right ear",   "left shoulder", "right shoulder", "left elbow",
    "right elbow", "left wrist",    "right wrist",    "left hip",
    "right hip",   "left knee",     "right knee",     "left ankle",
    "right ankle"};
static const size_t NUM_KEYPOINTS = KEYPOINTS.size();

class Keypoint {
public:
  explicit Keypoint() {}
  explicit Keypoint(std::string keypoint, std::array<size_t, 2> yx)
      : keypoint_(keypoint), yx_(yx) {}
  explicit Keypoint(std::string keypoint, std::array<size_t, 2> yx, float score)
      : keypoint_(keypoint), yx_(yx), score_(score) {}

  std::array<size_t, 2> &yx() { return yx_; }
  std::array<size_t, 2> yx() const { return yx_; }
  std::optional<float> score() const { return score_; }

private:
  std::string keypoint_;
  std::array<size_t, 2> yx_;
  std::optional<float> score_;
};

class Pose {
public:
  explicit Pose(std::unordered_map<std::string, Keypoint> keypoints)
      : keypoints_(keypoints), score_() {}
  explicit Pose(std::unordered_map<std::string, Keypoint> keypoints,
                float score)
      : keypoints_(keypoints), score_(score) {}

  std::unordered_map<std::string, Keypoint> keypoints() const {
    return keypoints_;
  }
  std::optional<float> score() const { return score_; }

private:
  std::unordered_map<std::string, Keypoint> keypoints_;
  std::optional<float> score_;
};

class Engine {
public:
  explicit Engine(std::string model_path, bool mirror = false)
      : engine_(std::make_unique<coral::BasicEngine>(model_path)),
        input_tensor_shape_(engine_->get_input_tensor_shape()),
        output_offsets_(Engine::make_output_offsets(
            engine_->get_all_output_tensors_sizes())),
        mirror_(mirror_) {
    if (input_tensor_shape_.size() != 4) {
      throw std::exception();
    }
    if (input_tensor_shape_[0] != 1) {
      throw std::exception();
    }
    if (depth() != 3) {
      throw std::exception();
    }
  }

  int32_t height() const { return input_tensor_shape_[1]; }

  int32_t width() const { return input_tensor_shape_[2]; }

  int32_t depth() const { return input_tensor_shape_[3]; }

  template <typename T>
  std::pair<std::vector<Pose>, std::chrono::milliseconds>
  detect_poses(xt::xarray<T> img) {
    const size_t zero = 0;
    img = xt::pad(
        img,
        {{zero, std::max(zero, static_cast<size_t>(height()) - img.shape(0))},
         {zero, std::max(zero, static_cast<size_t>(width()) - img.shape(1))},
         {zero, zero}});

    // XXX: auto cast into a vector<T>
    std::vector<T> input;
    input.reserve(img.size());
    for (auto elem : xt::ravel(img)) {
      input.push_back(elem);
    }
    auto outputs = engine_->RunInference(input);
    std::chrono::milliseconds inf_time(
        static_cast<uint64_t>(engine_->get_inference_time()));

    std::vector<size_t> keypoints_shape = {
        outputs[0].size() / (NUM_KEYPOINTS * 2), NUM_KEYPOINTS, 2};
    auto keypoints = xt::adapt(outputs[0], keypoints_shape);

    std::vector<size_t> keypoints_scores_shape = {
        outputs[1].size() / NUM_KEYPOINTS, NUM_KEYPOINTS};
    auto keypoint_scores = xt::adapt(outputs[1], keypoints_scores_shape);
    auto pose_scores = outputs[2];
    auto nposes = static_cast<size_t>(outputs[3][0]);

    std::vector<Pose> poses;
    poses.reserve(nposes);

    for (size_t pose_i = 0; pose_i < nposes; ++pose_i) {
      std::unordered_map<std::string, Keypoint> keypoint_map(NUM_KEYPOINTS);

      xt::xarray<float> keypoint =
          xt::view(keypoints, pose_i, xt::all(), xt::all());

      size_t point_i = 0;
      for (auto point = xt::axis_begin(keypoint, 0);
           point != xt::axis_end(keypoint, 0); ++point, ++point_i) {
        auto y = (*point)[0];
        auto x = (*point)[1];
        Keypoint keypoint(KEYPOINTS[point_i],
                          std::array<size_t, 2>{{(size_t)y, (size_t)x}},
                          keypoint_scores(pose_i, point_i));
        if (mirror_) {
          keypoint.yx()[1] = width() - keypoint.yx()[1];
        }
        keypoint_map[KEYPOINTS[point_i]] = keypoint;
      }
      poses.emplace_back(keypoint_map, pose_scores[pose_i]);
    }

    return std::make_pair(poses, inf_time);
  }

private:
  static std::vector<size_t>
  make_output_offsets(std::vector<size_t> output_sizes) {
    std::vector<size_t> result;
    result.reserve(output_sizes.size() + 1);
    result.push_back(0);

    size_t offset = 0;
    for (auto size : output_sizes) {
      offset += size;
      result.push_back(offset);
    }

    return result;
  }

private:
  std::unique_ptr<coral::BasicEngine> engine_;
  std::vector<int32_t> input_tensor_shape_;
  std::vector<size_t> output_offsets_;
  bool mirror_;
}; // namespace pose
} // namespace pose

int main(int argc, const char *argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Test load a model with coral::BasicEngine")(
      "path,p", po::value<std::string>(),
      "path to a Tensorflow Lite model")("num,n", po::value<size_t>());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }

  if (vm.count("path")) {
    auto model_path = vm["path"].as<std::string>();
    pose::Engine engine(model_path);
    auto arr = xt::load_image("./couple.jpg");

    size_t millis = 0;
    const size_t n = vm["num"].as<size_t>();
    for (size_t i = 0; i < n; ++i) {
      std::chrono::milliseconds inf_time;
      std::tie(std::ignore, inf_time) = engine.detect_poses(arr);
      millis += inf_time.count();
    }
    std::cout << (n / (millis / 1000.0)) << " fps" << std::endl;
    return 0;
  }

  return 1;
}
