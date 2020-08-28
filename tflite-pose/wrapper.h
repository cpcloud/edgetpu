#pragma once

#include "rust/cxx.h"

#include "edgetpu.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include "src/cpp/basic/basic_engine.h"
#include "src/cpp/posenet/posenet_decoder_op.h"

#include <iostream>
#include <memory>

namespace pose {

using coral::BasicEngine;

auto build_engine(rust::Str model_path) {
  return std::make_unique<BasicEngine>(std::string(model_path.data(), model_path.size()));
}

rust::Vec<float> run_inference(
    std::unique_ptr<BasicEngine> engine,
    rust::Slice<uint8_t> data,
) {
  std::vector<uint8_t> cpp_data(data.data(), data.data() + data.size());
  auto results = engine->RunInference(cpp_data);
  return rust::Vec(results[0]);
}

} // namespace pose
