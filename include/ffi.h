#pragma once

#include "coral/pipeline/common.h"
#include "rust/cxx.h"
#include <memory>
#include <vector>

struct TfLiteTensor;

namespace coral {

class PipelinedModelRunner;
class PipelineTensor;

} // namespace coral

namespace edgetpu {

class EdgeTpuContext;

} // namespace edgetpu

namespace tflite {

class Interpreter;
class FlatBufferModel;

} // namespace tflite

namespace internal {

class OutputTensor {
public:
  explicit OutputTensor(coral::PipelineTensor tensor,
                        std::shared_ptr<coral::PipelinedModelRunner> runner);
  ~OutputTensor();

private:
  coral::PipelineTensor tensor_;
  std::shared_ptr<coral::PipelinedModelRunner> runner_;
};

} // namespace internal
struct SegStats;
struct DeviceInfo;
enum class DeviceType : std::uint8_t;

void init_glog(rust::Str program_name);

std::shared_ptr<coral::PipelinedModelRunner> make_pipelined_model_runner(
    rust::Slice<const std::shared_ptr<tflite::Interpreter>> interpreters);

void set_pipelined_model_runner_input_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, std::size_t size);
void set_pipelined_model_runner_output_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, std::size_t size);

rust::Vec<std::size_t>
get_queue_sizes(const coral::PipelinedModelRunner &runner);

std::size_t get_input_queue_size(const coral::PipelinedModelRunner &runner);
std::size_t get_output_queue_size(const coral::PipelinedModelRunner &runner);

rust::Vec<float> dequantize_with_scale(const TfLiteTensor &tensor, float scale);

std::shared_ptr<edgetpu::EdgeTpuContext>
make_edge_tpu_context(DeviceType device_type, rust::Str device_path);

std::unique_ptr<tflite::FlatBufferModel> make_model(rust::Str model_path);

std::shared_ptr<tflite::Interpreter> make_interpreter_from_model(
    const tflite::FlatBufferModel &model,
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context,
    std::size_t num_threads);

bool push_input_tensor(std::shared_ptr<coral::PipelinedModelRunner> runner,
                       rust::Slice<const uint8_t> data);
bool push_input_tensor_empty(
    std::shared_ptr<coral::PipelinedModelRunner> runner);

std::unique_ptr<std::vector<internal::OutputTensor>>
pop_output_tensors(std::shared_ptr<coral::PipelinedModelRunner> runner,
                   bool &succeeded);

rust::Vec<DeviceInfo> get_all_device_infos();

std::size_t get_output_tensor_count(const tflite::Interpreter &interpreter);

const TfLiteTensor *get_output_tensor(const tflite::Interpreter &interpreter,
                                      std::size_t index);
const TfLiteTensor *get_input_tensor(const tflite::Interpreter &interpreter,
                                     std::size_t index);
