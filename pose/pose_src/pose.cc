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
#include <boost/range/adaptors.hpp>

#include <array>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "pose_constants.hpp"
#include "pose_decode.hpp"
#include "pose_engine.hpp"
#include "pose_types.hpp"

namespace po = boost::program_options;
using boost::adaptors::filtered;

struct HashPair {
  template <typename T> size_t operator()(const pose::Dims<T> &dim) const {
    size_t seed = 0;
    boost::hash_combine(seed, dim[0]);
    boost::hash_combine(seed, dim[1]);
    return seed;
  }
};

static const std::unordered_map<pose::Dims<size_t>, pose::Dims<size_t>,
                                HashPair>
    MODEL_DIMS_TO_IMAGE_DIMS = {{{641, 481}, {640, 480}},
                                {{1281, 721}, {1280, 720}},
                                {{481, 353}, {481, 353}},
                                {{416, 288}, {416, 288}}};

int main(int argc, const char *argv[]) {
  po::options_description desc("Allowed options");
  auto options = desc.add_options();
  options = options("help,h", "Display help message");
  options = options("model-path,m", po::value<std::string>(),
                    "path to a Tensorflow Lite model");
  options = options("device-path,d", po::value<std::string>(),
                    "path to a v4l2 compatible device");
  options = options("threshold,t", po::value<float>()->default_value(0.2f),
                    "pose keypoint score threshold");
  options = options("nms-radius,r", po::value<size_t>()->default_value(20));
  options = options("min-pose-score,s", po::value<float>()->default_value(0.2));
  options = options("width,w", po::value<size_t>(), "Image width")(
      "height,H", po::value<size_t>(), "Image height");
  options = options("max-poses,p", po::value<size_t>()->default_value(10),
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
  const auto model_width = vm["width"].as<size_t>();
  const auto model_height = vm["height"].as<size_t>();
  const auto max_poses = vm["max-poses"].as<size_t>();
  const auto min_pose_score = vm["min-pose-score"].as<float>();
  const auto nms_radius = vm["nms-radius"].as<size_t>();

  cv::VideoCapture capture(vm["device-path"].as<std::string>(), cv::CAP_V4L2);
  const auto [image_width, image_height] =
      MODEL_DIMS_TO_IMAGE_DIMS.at({model_width, model_height});
  capture.set(cv::CAP_PROP_FRAME_WIDTH, image_width);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, image_height);

  cv::Mat in_frame;
  cv::Mat out_frame;

  const cv::Size model_frame_dims{
      {static_cast<int32_t>(model_width), static_cast<int32_t>(model_height)}};

  pose::Seconds<float> inf_secs(0.0f);
  pose::Seconds<float> cam_secs(0.0f);
  pose::Seconds<float> resize_secs(0.0f);

  size_t nframes = 0;

  pose::Engine engine(model_path);

  while (true) {
    // const auto start_frame = std::chrono::steady_clock::now();
    // // in_frame = cv::imread(
    // //     "/home/cloud/code/edge/edgetpu/couple.jpg");
    // capture.read(in_frame);
    // const auto stop_frame = std::chrono::steady_clock::now();
    // cam_secs += stop_frame - start_frame;
    // ++nframes;
    //
    // const auto start_resize = std::chrono::steady_clock::now();
    // cv::resize(in_frame, out_frame, model_frame_dims, 0, 0,
    //            cv::InterpolationFlags::INTER_LINEAR);
    // const auto stop_resize = std::chrono::steady_clock::now();
    // resize_secs += stop_resize - start_resize;
    //
    // const size_t nbytes = out_frame.step[0] * out_frame.rows;
    // std::vector input(out_frame.data, out_frame.data + nbytes);
    //
    // const auto [poses, inf_millis] = engine.detect_poses(
    //     input, max_poses, threshold, nms_radius, min_pose_score);
    // inf_secs += inf_millis;
    // auto true_secs = cam_secs + resize_secs + inf_secs;
    //
    // for (const auto pose : poses) {
    //   std::array<std::optional<cv::Point2f>, pose::constants::NUM_KEYPOINTS>
    //       xys;
    //
    //   for (const auto keypoint :
    //        pose.keypoints() | filtered([threshold](const auto &keypoint) {
    //          return keypoint.score() >= threshold;
    //        })) {
    //     const auto point = keypoint.point();
    //     const cv::Point2f cv_point(point.x, point.y);
    //     xys[keypoint.index()] = cv_point;
    //     cv::circle(out_frame, cv_point, 6, cv::Scalar(0, 255, 0), -1);
    //   }
    //
    //   for (const auto [a, b] : pose::constants::PARENT_CHILD_TUPLES) {
    //     const auto axy = xys[a];
    //     const auto bxy = xys[b];
    //     if (axy.has_value() && bxy.has_value()) {
    //       cv::line(out_frame, axy.value(), bxy.value(), cv::Scalar(0, 255,
    //       255),
    //                2);
    //     }
    //   }
    // }
    //
    // const auto frames_text = boost::format("Frame: %d") % nframes;
    // cv::putText(out_frame, frames_text.str(), cv::Point(model_width / 2, 15),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
    //             cv::LINE_AA);
    //
    // const auto model_fps = nframes / inf_secs.count();
    // const auto model_fps_text = boost::format("Model FPS: %.1f") % model_fps;
    // cv::putText(out_frame, model_fps_text.str(), cv::Point(model_width / 2,
    // 30),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
    //             cv::LINE_AA);
    //
    // const auto cam_fps = nframes / cam_secs.count();
    // const auto cam_fps_text = boost::format("Cam FPS: %.1f") % cam_fps;
    // cv::putText(out_frame, cam_fps_text.str(), cv::Point(model_width / 2,
    // 45),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
    //             cv::LINE_AA);
    //
    // const auto resize_fps = nframes / resize_secs.count();
    // const auto resize_fps_text = boost::format("Resize FPS: %.1f") %
    // resize_fps; cv::putText(out_frame, resize_fps_text.str(),
    //             cv::Point(model_width / 2, 60), cv::FONT_HERSHEY_SIMPLEX,
    //             0.5, cv::Scalar(38, 0, 255), 1, cv::LINE_AA);
    //
    // const auto true_fps = nframes / true_secs.count();
    // const auto true_fps_text = boost::format("True FPS: %.1f") % true_fps;
    // cv::putText(out_frame, true_fps_text.str(), cv::Point(model_width / 2,
    // 75),
    //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(38, 0, 255), 1,
    //             cv::LINE_AA);

    // cv::imshow("Pose", out_frame);
    // if (cv::waitKey(5) > 0) {
    //   break;
    // }
  }

  return 0;
}
