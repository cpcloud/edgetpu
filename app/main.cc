#include "src/cpp/basic/basic_engine.h"

#include "xtensor/xadapt.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xaxis_iterator.hpp"
#include "xtensor/xaxis_slice_iterator.hpp"
#include "xtensor/xview.hpp"

#include <boost/container_hash/hash.hpp>
#include <boost/format.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <array>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "tinyformat.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "pose_constants.hpp"
#include "pose_decode.hpp"
#include "pose_engine.hpp"

namespace po = boost::program_options;

using Dims = std::pair<size_t, size_t>;

struct HashPair {
  template <typename T, typename U>
  size_t operator()(const std::pair<T, U> &pair) const {
    size_t seed = 0;
    boost::hash_combine(seed, std::get<0>(pair));
    boost::hash_combine(seed, std::get<1>(pair));
    return seed;
  }
};

static const std::unordered_map<Dims, Dims, HashPair> MODEL_DIMS_TO_IMAGE_DIMS =
    {{std::make_pair(641, 481), std::make_pair(640, 480)},
     {std::make_pair(1281, 721), std::make_pair(1280, 720)},
     {std::make_pair(481, 353), std::make_pair(481, 353)},
     {std::make_pair(416, 288), std::make_pair(416, 288)}};

int main(int argc, const char *argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Display help message")(
      "model-path,m", po::value<std::string>(),
      "path to a Tensorflow Lite model")("device-path,d",
                                         po::value<std::string>(),
                                         "path to a v4l2 compatible device")(
      "threshold,t", po::value<float>()->default_value(0.2f),
      "pose keypoint score threshold")("nms-radius,r",
                                       po::value<size_t>()->default_value(20))(
      "min-pose-score,s", po::value<float>()->default_value(0.2))(
      "width,w", po::value<int32_t>(),
      "Image width")("height,H", po::value<int32_t>(), "Image height")(
      "max-poses,p", po::value<size_t>()->default_value(10),
      "The maximum number of poses to detect in an image");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.empty() || vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  const auto model_path = vm["model-path"].as<std::string>();
  const auto threshold = vm["threshold"].as<float>();
  const auto model_width = vm["width"].as<int32_t>();
  const auto model_height = vm["height"].as<int32_t>();
  const auto max_poses = vm["max-poses"].as<size_t>();
  const auto min_pose_score = vm["min-pose-score"].as<float>();
  const auto nms_radius = vm["nms-radius"].as<size_t>();

  cv::VideoCapture capture(vm["device-path"].as<std::string>(), cv::CAP_V4L2);
  const auto [image_width, image_height] =
      MODEL_DIMS_TO_IMAGE_DIMS.at(std::make_pair(model_width, model_height));
  capture.set(cv::CAP_PROP_FRAME_WIDTH, image_width);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, image_height);

  cv::Mat in_frame;
  cv::Mat out_frame;

  const cv::Size model_frame_dims{{model_width, model_height}};

  pose::Seconds<float> inf_secs(0.0f);
  pose::Seconds<float> cam_secs(0.0f);
  pose::Seconds<float> resize_secs(0.0f);

  size_t nframes = 0;

  pose::Engine engine(model_path);

  for (;;) {
    const auto start_frame = std::chrono::steady_clock::now();
    in_frame = cv::imread(
        "/home/cloud/code/edge/edgetpu/couple.jpg"); // capture.read(in_frame);
    const auto stop_frame = std::chrono::steady_clock::now();
    cam_secs += stop_frame - start_frame;
    ++nframes;

    const auto start_resize = std::chrono::steady_clock::now();
    cv::resize(in_frame, out_frame, model_frame_dims, 0, 0,
               cv::InterpolationFlags::INTER_LINEAR);
    const auto stop_resize = std::chrono::steady_clock::now();
    resize_secs += stop_resize - start_resize;

    const size_t nbytes = out_frame.step[0] * out_frame.rows;
    std::vector input(out_frame.data, out_frame.data + nbytes);

    const auto [poses, inf_millis] =
        engine.detect_poses(input, max_poses, threshold, 1, min_pose_score);
    inf_secs += inf_millis;
    auto true_secs = cam_secs + resize_secs + inf_secs;

    for (const auto pose : poses) {
      std::array<std::optional<cv::Point2f>, pose::constants::NUM_KEYPOINTS>
          xys;

      for (const auto keypoint : pose.keypoints()) {
        if (keypoint.score() >= threshold) {
          const auto point = keypoint.point();
          const cv::Point2f cv_point(point.x, point.y);
          xys[keypoint.index()] = cv_point;
          tfm::printf("score: %s, point: %s\n", keypoint.score(), cv_point);
          cv::circle(out_frame, cv_point, 6, cv::Scalar(0, 255, 0), -1);
        }
      }

      for (const auto [a, b] : pose::constants::PARENT_CHILD_TUPLES) {
        const auto axy = xys[a];
        const auto bxy = xys[b];
        if (axy.has_value() && bxy.has_value()) {
          cv::line(out_frame, axy.value(), bxy.value(), cv::Scalar(0, 255, 255),
                   2);
        }
      }
    }

    const auto frames_text = boost::format("Frame: %d") % nframes;
    cv::putText(out_frame, frames_text.str(), cv::Point(model_width / 2, 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    const auto model_fps_text =
        boost::format("Model FPS: %.1f") % (nframes / inf_secs.count());
    cv::putText(out_frame, model_fps_text.str(), cv::Point(model_width / 2, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    const auto cam_fps_text =
        boost::format("Cam FPS: %.1f") % (nframes / cam_secs.count());
    cv::putText(out_frame, cam_fps_text.str(), cv::Point(model_width / 2, 45),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    const auto resize_fps_text =
        boost::format("Resize FPS: %.1f") % (nframes / resize_secs.count());
    cv::putText(out_frame, resize_fps_text.str(),
                cv::Point(model_width / 2, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(38, 0, 255), 1, cv::LINE_AA);

    const auto true_fps_text =
        boost::format("True FPS: %.1f") % (nframes / true_secs.count());
    cv::putText(out_frame, true_fps_text.str(), cv::Point(model_width / 2, 75),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
                cv::LINE_AA);

    cv::imshow("Pose", out_frame);
    if (cv::waitKey(5) > 0) {
      break;
    }
  }

  return 0;
}
