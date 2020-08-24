#pragma once

#include "rust/cxx.h"

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#include <iostream>
#include <memory>

namespace pose {

using tflite::FlatBufferModel;
using tflite::Interpreter;

auto build_model_from_file(rust::Str model_path) {
  const std::string path(model_path.data(), model_path.size());
  return FlatBufferModel::BuildFromFile(path.data());
}

rust::String input_name(const Interpreter &interp, size_t index) {
  return interp.tensor(index)->name;
}

auto num_inputs(const Interpreter &interp) { return interp.inputs().size(); }

std::unique_ptr<Interpreter> build_interpreter(const FlatBufferModel &model) {
  auto edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

} // namespace pose
