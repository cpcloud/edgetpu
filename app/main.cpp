#include "edgetpu.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "src/cpp/basic/basic_engine.h"

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()("help", "help message")("path", po::value<std::string>(),
                                             "path to a Tensorflow Lite model");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  if (vm.count("path")) {
    auto path = vm["path"].as<std::string>();
    coral::BasicEngine engine(path);
    return 0;
  } else {
    return 1;
  }
}
