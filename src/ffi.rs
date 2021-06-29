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

    unsafe extern "C++" {
        include!("coral/pipeline/pipelined_model_runner.h");
        include!("coral/pipeline/common.h");
        include!("tensorflow/lite/interpreter.h");
        include!("tensorflow/lite/model.h");
        include!("tflite/public/edgetpu.h");
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
        ) -> SharedPtr<PipelinedModelRunner>;

        fn set_pipelined_model_runner_input_queue_size(
            runner: SharedPtr<PipelinedModelRunner>,
            size: usize,
        );

        fn set_pipelined_model_runner_output_queue_size(
            runner: SharedPtr<PipelinedModelRunner>,
            size: usize,
        );

        fn push_input_tensors(
            runner: SharedPtr<PipelinedModelRunner>,
            inputs: &mut [SharedPtr<PipelineTensor>],
        ) -> Result<bool>;

        fn pop_output_tensors(
            runner: SharedPtr<PipelinedModelRunner>,
            outputs: &mut [UniquePtr<OutputTensor>],
        ) -> bool;

        fn make_input_tensor(
            runner: SharedPtr<PipelinedModelRunner>,
            data: &[u8],
        ) -> SharedPtr<PipelineTensor>;

        fn get_queue_sizes(runner: &PipelinedModelRunner) -> Vec<usize>;

        fn make_model(model_path: &str) -> UniquePtr<FlatBufferModel>;

        fn make_interpreter_from_model(
            model: &FlatBufferModel,
            edgetpu_context: SharedPtr<EdgeTpuContext>,
        ) -> Result<SharedPtr<Interpreter>>;

        fn make_edge_tpu_context(
            device_type: DeviceType,
            device_path: &str,
        ) -> SharedPtr<EdgeTpuContext>;

        fn get_all_device_infos() -> Vec<DeviceInfo>;

        fn get_output_tensor_count(interpreter: &Interpreter) -> usize;

        fn get_output_tensor(interpreter: &Interpreter, index: usize) -> *const TfLiteTensor;
    }
}
