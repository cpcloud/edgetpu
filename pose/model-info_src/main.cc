#include <fstream>
#include <iostream>

#include "flatbuffers/flexbuffers.h"
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <fmt/core.h>

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {
  po::options_description desc("Show information about a pose model");
  auto options = desc.add_options();
  options = options("help,h", "Display help message");
  options = options("model-path,m", po::value<std::string>(),
                    "path to a Tensorflow Lite model");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.empty() || vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }
  const auto model_path = vm["model-path"].as<std::string>();

  // open the file:
  std::ifstream file(model_path, std::ios::binary);

  // read the data:
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  const auto &m = flexbuffers::GetRoot(data.data(), data.size()).AsMap();

  const auto max_detections = m["max_detections"].AsInt32();
  const auto score_threshold = m["score_threshold"].AsFloat();
  const auto stride = m["stride"].AsInt32();
  const auto nms_radius = m["nms_radius"].AsFloat();

  fmt::print("max_detections:  {}\n"
             "score_threshold: {}\n"
             "stride:          {}\n"
             "nms_radius:      {}\n",
             max_detections, score_threshold, stride, nms_radius);
  return 0;
}
