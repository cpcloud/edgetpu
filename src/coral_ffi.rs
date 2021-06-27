#[cxx::bridge(namespace = "coral")]
mod ffi {
    struct SegStats {
        total_time_ns: i64,
        num_inferences: u64,
    }

    unsafe extern "C++" {
        include!("coral/pipeline/pipelined_model_runner.h");
        include!("coral/pipeline/common.h");
        include!("tflite-pose/include/coral_ffi.h");

        type PipelinedModelRunner;
        type PipelineTensor;
        type Interpreter = crate::tf::Interpreter;

        fn make_pipelined_model_runner(
            interpreters: &[*mut Interpreter],
        ) -> UniquePtr<PipelinedModelRunner>;
        // fn push_tensors(
        //     runner: UniquePtr<PipelinedModelRunner>,
        //     inputs: Vec<UniquePtr<PipelineTensor>>,
        // );
        // fn pop_tensors(runner: UniquePtr<PipelinedModelRunner>) -> Vec<UniquePtr<PipelineTensor>>;
        fn get_segment_stats(runner: UniquePtr<PipelinedModelRunner>) -> Vec<SegStats>;
        fn get_queue_sizes(runner: UniquePtr<PipelinedModelRunner>) -> Vec<usize>;
    }
}
