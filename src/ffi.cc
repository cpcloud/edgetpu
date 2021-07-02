#include "tflite-pose/include/ffi.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/tflite_utils.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite-pose/src/ffi.rs.h"
#include "tflite/public/edgetpu.h"

#include <algorithm>
#include <memory>

namespace {

edgetpu::DeviceType device_type_to_edgetpu_device_type(DeviceType device_type) {
  switch (device_type) {
  case DeviceType::Pci:
    return edgetpu::DeviceType::kApexPci;
  case DeviceType::Usb:
    return edgetpu::DeviceType::kApexUsb;
  default:
    assert(false);
  }
}

DeviceType edgetpu_device_type_to_device_type(edgetpu::DeviceType device_type) {
  switch (device_type) {
  case edgetpu::DeviceType::kApexPci:
    return DeviceType::Pci;
  case edgetpu::DeviceType::kApexUsb:
    return DeviceType::Usb;
  default:
    assert(false);
  }
}
} // namespace

rust::Vec<std::size_t>
get_queue_sizes(const coral::PipelinedModelRunner &runner) {
  const auto queue_sizes = runner.GetQueueSizes();

  rust::Vec<std::size_t> results;
  results.reserve(queue_sizes.size());

  for (const auto &size : queue_sizes) {
    results.push_back(size);
  }
  return results;
}

std::shared_ptr<coral::PipelinedModelRunner> make_pipelined_model_runner(
    rust::Slice<const std::shared_ptr<tflite::Interpreter>> interpreters) {
  std::vector<tflite::Interpreter *> interps;
  interps.reserve(interpreters.size());
  for (auto &interp : interpreters) {
    interps.push_back(interp.get());
  }
  return std::make_shared<coral::PipelinedModelRunner>(interps, nullptr,
                                                       nullptr);
}

void set_pipelined_model_runner_input_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, size_t size) {
  runner->SetInputQueueSize(size);
}

void set_pipelined_model_runner_output_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, size_t size) {
  runner->SetOutputQueueSize(size);
}

std::shared_ptr<edgetpu::EdgeTpuContext>
make_edge_tpu_context(DeviceType device_type, rust::Str device_path) {
  return edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
      device_type_to_edgetpu_device_type(device_type),
      static_cast<std::string>(device_path));
}

std::unique_ptr<tflite::FlatBufferModel> make_model(rust::Str model_path) {
  auto path = static_cast<std::string>(model_path);
  return tflite::FlatBufferModel::BuildFromFile(path.c_str());
}

std::shared_ptr<tflite::Interpreter> make_interpreter_from_model(
    const tflite::FlatBufferModel &model,
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context,
    size_t num_threads) {
  auto interpreter =
      coral::MakeEdgeTpuInterpreterOrDie(model, edgetpu_context.get());
  interpreter->SetNumThreads(num_threads);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    throw std::logic_error("failed to allocate tensors");
  }
  return interpreter;
}

namespace internal {

OutputTensor::OutputTensor(std::unique_ptr<coral::PipelineTensor> tensor,
                           std::shared_ptr<coral::PipelinedModelRunner> runner)
    : tensor_(std::move(tensor)), runner_(runner) {}
OutputTensor::~OutputTensor() {
  runner_->GetOutputTensorAllocator()->Free(tensor_->buffer);
}

} // namespace internal

std::shared_ptr<coral::PipelineTensor>
make_input_tensor(std::shared_ptr<coral::PipelinedModelRunner> runner,
                  rust::Slice<const uint8_t> data) {
  auto buffer = runner->GetInputTensorAllocator()->Alloc(data.size());
  std::copy(data.begin(), data.end(),
            reinterpret_cast<uint8_t *>(buffer->ptr()));
  return std::shared_ptr<coral::PipelineTensor>(
      new coral::PipelineTensor{.type = TfLiteType::kTfLiteUInt8,
                                .buffer = buffer,
                                .bytes = data.size()});
}

bool push_input_tensors(
    std::shared_ptr<coral::PipelinedModelRunner> runner,
    rust::Slice<std::shared_ptr<coral::PipelineTensor>> inputs) {
  std::vector<coral::PipelineTensor> cpp_inputs;
  cpp_inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    cpp_inputs.push_back(*inputs[i]);
  }
  return runner->Push(cpp_inputs);
}

bool pop_output_tensors(
    std::shared_ptr<coral::PipelinedModelRunner> runner,
    rust::Slice<std::unique_ptr<internal::OutputTensor>> outputs) {
  std::vector<coral::PipelineTensor> raw_outputs;
  auto result = runner->Pop(&raw_outputs);

  for (size_t i = 0; i < raw_outputs.size(); ++i) {
    outputs[i] = std::make_unique<internal::OutputTensor>(
        std::make_unique<coral::PipelineTensor>(raw_outputs[i]), runner);
  }

  return result;
}

rust::Vec<DeviceInfo> get_all_device_infos() {
  rust::Vec<DeviceInfo> device_infos;
  for (auto record :
       edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu()) {
    device_infos.push_back(
        DeviceInfo{edgetpu_device_type_to_device_type(record.type),
                   rust::String(record.path)});
  }
  return device_infos;
}

std::size_t get_output_tensor_count(const tflite::Interpreter &interpreter) {
  return interpreter.outputs().size();
}

const TfLiteTensor *get_output_tensor(const tflite::Interpreter &interpreter,
                                      size_t index) {
  return interpreter.output_tensor(index);
}

const TfLiteTensor *get_input_tensor(const tflite::Interpreter &interpreter,
                                     size_t index) {
  return interpreter.input_tensor(index);
}
