#include "tflite-pose/include/ffi.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "coral/tflite_utils.h"
#include "glog/logging.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite-pose/src/ffi.rs.h"
#include "tflite/public/edgetpu.h"

#include <algorithm>
#include <iterator>
#include <memory>

using google::ERROR;

namespace {

edgetpu::DeviceType device_type_to_edgetpu_device_type(DeviceType device_type) {
  switch (device_type) {
  case DeviceType::Pci:
    return edgetpu::DeviceType::kApexPci;
  case DeviceType::Usb:
    return edgetpu::DeviceType::kApexUsb;
  default:
    throw std::invalid_argument("invalid DeviceType");
  }
}

DeviceType edgetpu_device_type_to_device_type(edgetpu::DeviceType device_type) {
  switch (device_type) {
  case edgetpu::DeviceType::kApexPci:
    return DeviceType::Pci;
  case edgetpu::DeviceType::kApexUsb:
    return DeviceType::Usb;
  default:
    throw std::invalid_argument("invalid edgetpu::DeviceType");
  }
}
} // namespace

rust::Vec<std::size_t>
get_queue_sizes(const coral::PipelinedModelRunner &runner) {
  const auto queue_sizes = runner.GetQueueSizes();

  rust::Vec<std::size_t> results;
  results.reserve(queue_sizes.size());

  std::copy(queue_sizes.cbegin(), queue_sizes.cend(),
            std::back_insert_iterator(results));

  return results;
}

std::size_t get_input_queue_size(const coral::PipelinedModelRunner &runner) {
  return runner.GetInputQueueSize();
}

std::size_t get_output_queue_size(const coral::PipelinedModelRunner &runner) {
  return runner.GetOutputQueueSize();
}

std::shared_ptr<coral::PipelinedModelRunner> make_pipelined_model_runner(
    rust::Slice<const std::shared_ptr<tflite::Interpreter>> interpreters) {
  std::vector<tflite::Interpreter *> interps;
  interps.reserve(interpreters.size());
  std::transform(interpreters.begin(), interpreters.end(),
                 std::back_insert_iterator(interps), [](auto interp) {
                   if (interp == nullptr) {
                     throw std::invalid_argument("interpreter is nullptr");
                   }
                   return interp.get();
                 });
  return std::make_shared<coral::PipelinedModelRunner>(interps);
}

void set_pipelined_model_runner_input_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, std::size_t size) {
  if (runner == nullptr) {
    throw std::invalid_argument(
        "runner is nullptr when setting input queue size");
  }
  runner->SetInputQueueSize(size);
}

void set_pipelined_model_runner_output_queue_size(
    std::shared_ptr<coral::PipelinedModelRunner> runner, std::size_t size) {
  if (runner == nullptr) {
    throw std::invalid_argument(
        "runner is nullptr when setting output queue size");
  }
  runner->SetOutputQueueSize(size);
}

std::shared_ptr<edgetpu::EdgeTpuContext>
make_edge_tpu_context(DeviceType device_type, rust::Str device_path) {
  auto edgetpu_singleton = edgetpu::EdgeTpuManager::GetSingleton();
  if (edgetpu_singleton == nullptr) {
    throw std::logic_error("edgetpu_singleton is nullptr");
  }
  auto device = edgetpu_singleton->OpenDevice(
      device_type_to_edgetpu_device_type(device_type),
      static_cast<std::string>(device_path));
  if (device == nullptr) {
    throw std::logic_error("edgetpu device is nullptr");
  }
  return device;
}

std::unique_ptr<tflite::FlatBufferModel> make_model(rust::Str model_path) {
  auto path = static_cast<std::string>(model_path);
  auto model = tflite::FlatBufferModel::BuildFromFile(path.c_str());

  if (model == nullptr) {
    throw std::logic_error("tflite flatbuffer model is nullptr");
  }

  return model;
}

std::shared_ptr<tflite::Interpreter> make_interpreter_from_model(
    const tflite::FlatBufferModel &model,
    std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context,
    std::size_t num_threads) {
  auto context = edgetpu_context.get();
  if (context == nullptr) {
    throw std::invalid_argument("edgetpu_context is nullptr");
  }

  auto interpreter = coral::MakeEdgeTpuInterpreterOrDie(model, context);
  if (interpreter == nullptr) {
    throw std::logic_error("edgetpu interpreter is nullptr");
  }

  interpreter->SetNumThreads(num_threads);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    throw std::logic_error("failed to allocate tensors");
  }

  return interpreter;
}

namespace internal {

OutputTensor::OutputTensor(std::unique_ptr<coral::PipelineTensor> tensor,
                           std::shared_ptr<coral::PipelinedModelRunner> runner)
    : tensor_(std::move(tensor)), runner_(runner) {
  if (tensor_ == nullptr) {
    throw std::invalid_argument("tensor_ is nullptr");
  }

  if (runner_ == nullptr) {
    throw std::invalid_argument("runner_ is nullptr");
  }
}

OutputTensor::~OutputTensor() {
  if (runner_ == nullptr) {
    rust_log_error("runner_ is nullptr");
    return;
  }

  auto *output_tensor_allocator = runner_->GetOutputTensorAllocator();

  if (output_tensor_allocator == nullptr) {
    rust_log_error("output_tensor_allocator is nullptr");
    return;
  }

  if (tensor_ == nullptr) {
    rust_log_error("tensor_ is nullptr");
    return;
  }

  output_tensor_allocator->Free(tensor_->buffer);
}

} // namespace internal

void init_glog(rust::Str program_name) {
  const auto prog_name = static_cast<std::string>(program_name);
  google::InitGoogleLogging(prog_name.c_str());
}

std::shared_ptr<coral::PipelineTensor>
make_input_tensor(std::shared_ptr<coral::PipelinedModelRunner> runner,
                  rust::Slice<const uint8_t> data) {
  if (runner == nullptr) {
    throw std::logic_error("runner is nullptr");
  }

  auto input_tensor_allocator = runner->GetInputTensorAllocator();

  if (input_tensor_allocator == nullptr) {
    throw std::logic_error("input_tensor_allocator is nullptr");
  }

  auto buffer = input_tensor_allocator->Alloc(data.size());

  if (buffer == nullptr) {
    throw std::logic_error("input buffer is nullptr");
  }

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
  if (runner == nullptr) {
    throw std::logic_error("runner is nullptr");
  }

  std::vector<coral::PipelineTensor> cpp_inputs;
  cpp_inputs.reserve(inputs.size());

  std::transform(inputs.begin(), inputs.end(),
                 std::back_insert_iterator(cpp_inputs),
                 [](auto input) { return *input; });

  return runner->Push(cpp_inputs);
}

bool pop_output_tensors(
    std::shared_ptr<coral::PipelinedModelRunner> runner,
    rust::Slice<std::unique_ptr<internal::OutputTensor>> outputs) {
  if (runner == nullptr) {
    throw std::logic_error("runner is nullptr");
  }

  std::vector<coral::PipelineTensor> raw_outputs;
  auto result = runner->Pop(&raw_outputs);

  std::transform(raw_outputs.cbegin(), raw_outputs.cend(), outputs.begin(),
                 [runner](auto raw_output) {
                   return std::make_unique<internal::OutputTensor>(
                       std::make_unique<coral::PipelineTensor>(raw_output),
                       runner);
                 });
  return result;
}

rust::Vec<DeviceInfo> get_all_device_infos() {
  auto edgetpu_singleton = edgetpu::EdgeTpuManager::GetSingleton();
  if (edgetpu_singleton == nullptr) {
    throw std::logic_error("edgetpu_singleton is nullptr");
  }

  auto enumerations = edgetpu_singleton->EnumerateEdgeTpu();

  rust::Vec<DeviceInfo> device_infos;
  device_infos.reserve(enumerations.size());

  std::transform(enumerations.cbegin(), enumerations.cend(),
                 std::back_insert_iterator(device_infos), [](auto record) {
                   return DeviceInfo{
                       edgetpu_device_type_to_device_type(record.type),
                       rust::String(record.path)};
                 });
  return device_infos;
}

std::size_t get_output_tensor_count(const tflite::Interpreter &interpreter) {
  return interpreter.outputs().size();
}

const TfLiteTensor *get_output_tensor(const tflite::Interpreter &interpreter,
                                      std::size_t index) {
  auto result = interpreter.output_tensor(index);
  if (result == nullptr) {
    throw std::logic_error("output tensor is nullptr");
  }
  return result;
}

const TfLiteTensor *get_input_tensor(const tflite::Interpreter &interpreter,
                                     std::size_t index) {
  auto result = interpreter.input_tensor(index);
  if (result == nullptr) {
    throw std::logic_error("input tensor is nullptr");
  }
  return result;
}
