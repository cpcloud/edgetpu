use crate::{error::Error, ffi::ffi, tflite::Interpreter};
use cxx::{SharedPtr, UniquePtr};
use std::sync::Arc;
use tracing::{debug, instrument};

#[derive(Clone)]
pub(crate) struct PipelinedModelRunner {
    runner: SharedPtr<ffi::PipelinedModelRunner>,
    interpreters: Box<[Interpreter]>,
}

unsafe impl Sync for PipelinedModelRunner {}
unsafe impl Send for PipelinedModelRunner {}

impl PipelinedModelRunner {
    pub(crate) fn new(mut interpreters: Vec<Interpreter>) -> Result<Self, Error> {
        if interpreters.is_empty() {
            return Err(Error::ConstructPipelineModelRunnerFromInterpreters);
        }

        let mut pointers = interpreters
            .iter_mut()
            .map(|interp| interp.raw())
            .collect::<Vec<_>>();

        if pointers.is_empty() {
            return Err(Error::GetInterpreterPointers);
        }

        Ok(Self {
            runner: ffi::make_pipelined_model_runner(&mut pointers),
            interpreters: interpreters.into_boxed_slice(),
        })
    }

    pub(crate) fn num_interpreters(&self) -> usize {
        self.interpreters.len()
    }

    pub(crate) fn output_interpreter(&self) -> Result<&Interpreter, Error> {
        self.interpreters.last().ok_or(Error::GetOutputInterpreter)
    }

    pub(crate) fn set_input_queue_size(&mut self, size: usize) {
        ffi::set_pipelined_model_runner_input_queue_size(self.runner.clone(), size)
    }

    pub(crate) fn set_output_queue_size(&mut self, size: usize) {
        ffi::set_pipelined_model_runner_output_queue_size(self.runner.clone(), size)
    }

    pub(crate) fn segment_stats(&self) -> Vec<ffi::SegStats> {
        ffi::get_segment_stats(self.runner.clone())
    }

    pub(crate) fn queue_sizes(&self) -> Vec<usize> {
        ffi::get_queue_sizes(self.runner.clone())
    }

    #[instrument(
        name = "PipelinedModelRunner::push",
        skip(self, tensors),
        level = "debug"
    )]
    pub(crate) fn push(&self, tensors: Option<Arc<Vec<InputTensor>>>) -> bool {
        let mut ptrs = tensors.map_or_else(Default::default, |tensors| {
            tensors
                .iter()
                .map(|tensor| tensor.raw())
                .collect::<Vec<_>>()
        });
        ffi::push_input_tensors(self.runner.clone(), &mut ptrs).unwrap()
    }

    #[instrument(name = "PipelinedModelRunner::pop", skip(self), level = "debug")]
    pub(crate) fn pop(&self) -> Result<Vec<UniquePtr<ffi::OutputTensor>>, Error> {
        let num_output_tensors = self.output_interpreter()?.get_output_tensor_count();
        let mut tensors = Vec::with_capacity(num_output_tensors);
        let succeeded = ffi::pop_output_tensors(self.runner.clone(), &mut tensors);
        unsafe {
            tensors.set_len(num_output_tensors);
        }

        if succeeded {
            debug!("popping from output queue {:?}", self.queue_sizes());
            Ok(tensors)
        } else {
            Err(Error::PopPipelinedModelOutputTensors)
        }
    }

    pub(crate) fn alloc_input_tensor(&self, data: &[u8]) -> Result<InputTensor, Error> {
        InputTensor::new(ffi::make_input_tensor(self.runner.clone(), data))
    }
}

/// The buffer inside these are owned by the allocator, so we do not implement Drop.
#[derive(Clone)]
pub(crate) struct InputTensor {
    tensor: SharedPtr<ffi::InputTensor>,
}

unsafe impl Send for InputTensor {}
unsafe impl Sync for InputTensor {}

impl InputTensor {
    fn new(tensor: SharedPtr<ffi::InputTensor>) -> Result<Self, Error> {
        Ok(Self { tensor })
    }

    fn raw(&self) -> SharedPtr<ffi::InputTensor> {
        self.tensor.clone()
    }
}
