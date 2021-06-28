use cxx::{type_id, ExternType};

unsafe impl ExternType for crate::tflite_sys::TfLiteTensor {
    type Id = type_id!("TfLiteTensor");
    type Kind = cxx::kind::Opaque;
}

#[cxx::bridge]
pub(crate) mod ffi {
    struct SegStats {
        total_time_ns: i64,
        num_inferences: u64,
    }

    enum DeviceType {
        Pci,
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
        include!("tflite-pose/include/coral_ffi.h");

        #[namespace = "tflite"]
        type Interpreter;
        #[namespace = "tflite"]
        type FlatBufferModel;

        #[namespace = "edgetpu"]
        type EdgeTpuContext;

        #[namespace = "coral"]
        type PipelinedModelRunner;

        #[namespace = "internal"]
        type Tensor;

        fn make_pipelined_model_runner(
            interpreters: &mut [SharedPtr<Interpreter>],
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
            inputs: &mut [SharedPtr<Tensor>],
        ) -> Result<bool>;

        fn pop_output_tensors(
            runner: SharedPtr<PipelinedModelRunner>,
            outputs: &mut [UniquePtr<Tensor>],
        ) -> bool;

        fn make_input_tensor(
            runner: SharedPtr<PipelinedModelRunner>,
            data: &[u8],
        ) -> SharedPtr<Tensor>;

        fn get_segment_stats(runner: SharedPtr<PipelinedModelRunner>) -> Vec<SegStats>;
        fn get_queue_sizes(runner: SharedPtr<PipelinedModelRunner>) -> Vec<usize>;

        fn make_model(model_path: &str) -> SharedPtr<FlatBufferModel>;

        fn make_interpreter_from_model(
            model: SharedPtr<FlatBufferModel>,
            edgetpu_context: SharedPtr<EdgeTpuContext>,
        ) -> Result<SharedPtr<Interpreter>>;

        fn make_edge_tpu_context(
            device_type: DeviceType,
            device_path: &str,
        ) -> SharedPtr<EdgeTpuContext>;

        fn get_all_device_infos() -> Vec<DeviceInfo>;

        fn get_output_tensor_count(interpreter: SharedPtr<Interpreter>) -> usize;

        type TfLiteTensor = crate::tflite_sys::TfLiteTensor;

        fn get_output_tensor(
            interpreter: SharedPtr<Interpreter>,
            index: usize,
        ) -> *mut TfLiteTensor;
    }
}
