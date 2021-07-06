#![allow(unreachable_pub)]

#[cxx::bridge]
pub(crate) mod ffi {
    struct SegStats {
        total_time_ns: i64,
        num_inferences: u64,
    }

    enum DeviceType {
        #[allow(dead_code)]
        Pci,
        #[allow(dead_code)]
        Usb,
    }

    struct DeviceInfo {
        typ: DeviceType,
        path: String,
    }

    extern "Rust" {
        fn rust_log_error(msg: &CxxString);
    }

    unsafe extern "C++" {
        include!("tensorflow/lite/interpreter.h");
        include!("tensorflow/lite/model.h");
        include!("tflite/public/edgetpu.h");
        include!("coral/pipeline/pipelined_model_runner.h");
        include!("coral/pipeline/common.h");
        include!("tflite-pose/include/ffi.h");

        #[namespace = "tflite"]
        type Interpreter;

        #[namespace = "tflite"]
        type FlatBufferModel;

        #[namespace = "edgetpu"]
        type EdgeTpuContext;

        #[namespace = "coral"]
        type PipelinedModelRunner;

        #[namespace = "coral"]
        type PipelineTensor;

        #[namespace = "internal"]
        type OutputTensor;

        type TfLiteTensor = crate::tflite_sys::TfLiteTensor;

        fn make_pipelined_model_runner(
            interpreters: &[SharedPtr<Interpreter>],
        ) -> Result<SharedPtr<PipelinedModelRunner>>;

        fn set_pipelined_model_runner_input_queue_size(
            runner: SharedPtr<PipelinedModelRunner>,
            size: usize,
        ) -> Result<()>;

        fn set_pipelined_model_runner_output_queue_size(
            runner: SharedPtr<PipelinedModelRunner>,
            size: usize,
        ) -> Result<()>;

        fn push_input_tensors(
            runner: SharedPtr<PipelinedModelRunner>,
            inputs: &mut [SharedPtr<PipelineTensor>],
        ) -> Result<bool>;

        fn pop_output_tensors(
            runner: SharedPtr<PipelinedModelRunner>,
            outputs: &mut [UniquePtr<OutputTensor>],
        ) -> Result<bool>;

        fn make_input_tensor(
            runner: SharedPtr<PipelinedModelRunner>,
            data: &[u8],
        ) -> Result<SharedPtr<PipelineTensor>>;

        fn get_queue_sizes(runner: &PipelinedModelRunner) -> Result<Vec<usize>>;

        fn get_input_queue_size(runner: &PipelinedModelRunner) -> Result<usize>;
        fn get_output_queue_size(runner: &PipelinedModelRunner) -> Result<usize>;

        fn make_model(model_path: &str) -> Result<UniquePtr<FlatBufferModel>>;

        fn make_interpreter_from_model(
            model: &FlatBufferModel,
            edgetpu_context: SharedPtr<EdgeTpuContext>,
            num_threads: usize,
        ) -> Result<SharedPtr<Interpreter>>;

        fn make_edge_tpu_context(
            device_type: DeviceType,
            device_path: &str,
        ) -> Result<SharedPtr<EdgeTpuContext>>;

        fn get_all_device_infos() -> Result<Vec<DeviceInfo>>;

        fn get_output_tensor_count(interpreter: &Interpreter) -> Result<usize>;

        fn get_output_tensor(
            interpreter: &Interpreter,
            index: usize,
        ) -> Result<*const TfLiteTensor>;

        fn get_input_tensor(interpreter: &Interpreter, index: usize)
            -> Result<*const TfLiteTensor>;

        fn init_glog(program_name: &str) -> Result<()>;
    }
}

fn rust_log_error(msg: &cxx::CxxString) {
    tracing::error!(message = msg.to_string_lossy().as_ref());
}
