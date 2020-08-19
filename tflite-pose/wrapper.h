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
  std::string path(model_path.data(), model_path.size());
  return FlatBufferModel::BuildFromFile(path.data());
}

auto allocate_tensors(Interpreter &interpreter) {
  interpreter.AllocateTensors();
}

auto build_interpreter(FlatBufferModel &model) {
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
