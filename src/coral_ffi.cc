#include "tflite-pose/include/coral_ffi.h"
#include "coral/pipeline/common.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/interpreter.h"
#include "tflite-pose/src/coral_ffi.rs.h"

#include <memory>
#include <tuple>

namespace coral {

rust::Vec<SegStats>
get_segment_stats(std::unique_ptr<PipelinedModelRunner> runner) {
  const auto segment_stats = runner->GetSegmentStats();
  rust::Vec<SegStats> results;
  results.reserve(segment_stats.size());
  for (const auto &stat : segment_stats) {
    results.push_back(SegStats{stat.total_time_ns, stat.num_inferences});
  }
  return results;
}

rust::Vec<std::size_t>
get_queue_sizes(std::unique_ptr<PipelinedModelRunner> runner) {
  const auto queue_sizes = runner->GetQueueSizes();
  rust::Vec<std::size_t> results;
  results.reserve(queue_sizes.size());
  for (const auto &size : queue_sizes) {
    results.push_back(size);
  }
  return results;
}

std::unique_ptr<PipelinedModelRunner>
make_pipelined_model_runner(rust::Slice<TfLiteInterpreter *> interpreters) {
  std::vector<tflite::Interpreter *> interps;
  interps.reserve(interpreters.size());
  for (const auto *interp : interpreters) {
    interps.push_back(interp->impl.get());
  }
  return std::make_unique<PipelinedModelRunner>(interps);
}

} // namespace coral

namespace tflite {
  std::unique_ptr<tflite::FlatBufferModel> make_model(rust::String path) {
    return tflite::FlatBufferModel::BuildFromFile(path.c_str());
  }

  std::unique_ptr<tflite::Interpreter> make_edgetpu_interpreter(std::unique_ptr<tflite::FlatBufferModel> model) {
  }

  std::unique_ptr<tflite::Interpreter> make_interpreter_from_path(rust::String path);
}
