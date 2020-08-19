#include "edgetpu.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace po = boost::program_options;

std::vector<uint8_t> decode_bmp(const uint8_t *input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
      case 1:
        output[dst_pos] = input[src_pos];
        break;
      case 3:
        // BGR -> RGB
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        break;
      case 4:
        // BGRA -> RGBA
        output[dst_pos] = input[src_pos + 2];
        output[dst_pos + 1] = input[src_pos + 1];
        output[dst_pos + 2] = input[src_pos];
        output[dst_pos + 3] = input[src_pos + 3];
        break;
      default:
        std::cerr << "Unexpected number of channels: " << channels << std::endl;
        std::abort();
        break;
      }
    }
  }
  return output;
}

std::unique_ptr<tflite::Interpreter>
BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model,
                        edgetpu::EdgeTpuContext *edgetpu_context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

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
    auto model = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    auto edgetpu_context =
        edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    auto model_interpreter =
        BuildEdgeTpuInterpreter(*model, edgetpu_context.get());
    return 0;
  } else {
    std::cout << "path argument required" << std::endl;
    return 1;
  }
}
