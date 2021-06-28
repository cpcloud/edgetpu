#pragma once

#include "rust/cxx.h"
#include "coral/pipeline/common.h"

#include <memory>

namespace coral {
  class PipelinedModelRunner;
} // namespace coral

namespace edgetpu {

class EdgeTpuContext;

} // namespace edgetpu

namespace tflite {

class Interpreter;
class FlatBufferModel;

} // namespace tflite

namespace internal {

class InputTensor {
public:
  explicit InputTensor(coral::PipelineTensor tensor,
                       std::shared_ptr<coral::PipelinedModelRunner> runner);
  ~InputTensor() = default;
  coral::PipelineTensor tensor();

private:
  coral::PipelineTensor tensor_;
  std::shared_ptr<coral::PipelinedModelRunner> runner_;
};

class OutputTensor {
public:
  explicit OutputTensor(coral::PipelineTensor tensor,
                        std::shared_ptr<coral::PipelinedModelRunner> runner);
  ~OutputTensor();
  coral::PipelineTensor tensor();

private:
  coral::PipelineTensor tensor_;
  std::shared_ptr<coral::PipelinedModelRunner> runner_;
};

} // namespace internal
struct SegStats;
struct DeviceInfo;
enum class DeviceType : uint8_t;

std::shared_ptr<coral::PipelinedModelRunner>
make_pipelined_model_runner(rust::Slice<std::shared_ptr<tflite::Interpreter>> interpreters);

void set_pipelined_model_runner_input_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, size_t size);
void set_pipelined_model_runner_output_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, size_t size);

rust::Vec<SegStats>
get_segment_stats(std::shared_ptr<coral::PipelinedModelRunner> runner);

rust::Vec<std::size_t>
get_queue_sizes(std::shared_ptr<coral::PipelinedModelRunner> runner);

std::shared_ptr<edgetpu::EdgeTpuContext>
make_edge_tpu_context(DeviceType device_type, rust::Str device_path);

std::shared_ptr<tflite::FlatBufferModel> make_model(rust::Str model_path);

std::shared_ptr<tflite::Interpreter> make_interpreter_from_model(
    std::shared_ptr<tflite::FlatBufferModel> model,
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context);

std::shared_ptr<internal::InputTensor>
make_input_tensor(std::shared_ptr<coral::PipelinedModelRunner> runner,
                  rust::Slice<const uint8_t> data);

bool push_input_tensors(std::shared_ptr<coral::PipelinedModelRunner> runner,
        rust::Slice<std::shared_ptr<internal::InputTensor>> inputs);

bool
pop_output_tensors(std::shared_ptr<coral::PipelinedModelRunner> runner,
        rust::Slice<std::unique_ptr<internal::OutputTensor>> outputs);

rust::Vec<DeviceInfo> get_all_device_infos();

std::size_t get_output_tensor_count(std::shared_ptr<tflite::Interpreter> interpreter);
TfLiteTensor* get_output_tensor(std::shared_ptr<tflite::Interpreter> interpreter, size_t index);
