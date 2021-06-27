#pragma once

#include "rust/cxx.h"
#include <memory>

struct TfLiteInterpreter;

namespace coral {

struct SegStats;
class PipelinedModelRunner;

std::unique_ptr<PipelinedModelRunner>
make_pipelined_model_runner(rust::Slice<TfLiteInterpreter *> interpreters);

rust::Vec<SegStats>
get_segment_stats(std::unique_ptr<PipelinedModelRunner> runner);

rust::Vec<std::size_t>
get_queue_sizes(std::unique_ptr<PipelinedModelRunner> runner);

} // namespace coral

namespace tflite {
  std::unique_ptr<tflite::Model> make_model(rust::String path);
  std::unique_ptr<tflite::Interpreter> make_interpreter(std::unique_ptr<tflite::Model> model);
  std::unique_ptr<tflite::Interpreter> make_interpreter_from_path(rust::String path);
}
