use crate::{error::Error, ffi::ffi, tflite::Interpreter};
use cxx::{SharedPtr, UniquePtr};
use std::sync::Arc;
use tracing::{debug, error, instrument};

#[derive(Clone)]
pub(crate) struct PipelinedModelRunner {
    runner: SharedPtr<ffi::PipelinedModelRunner>,
    interpreters: Arc<[Interpreter]>,
    output_tensor_count: usize,
}

unsafe impl Sync for PipelinedModelRunner {}
unsafe impl Send for PipelinedModelRunner {}

impl PipelinedModelRunner {
    pub(crate) fn new(mut interpreters: Vec<Interpreter>) -> Result<Self, Error> {
        if interpreters.is_empty() {
            return Err(Error::ConstructPipelineModelRunnerFromInterpreters);
        }

        let pointers = interpreters
            .iter_mut()
            .map(|interp| interp.as_inner())
            .collect::<Vec<_>>();

        if pointers.is_empty() {
            return Err(Error::GetInterpreterPointers);
        }

        let output_tensor_count = interpreters
            .last()
            .ok_or(Error::GetOutputInterpreter)?
            .get_output_tensor_count()?;

        Ok(Self {
            runner: ffi::make_pipelined_model_runner(&pointers)
                .map_err(Error::MakePipelinedModelRunner)?,
            interpreters: interpreters.into_boxed_slice().into(),
            output_tensor_count,
        })
    }

    pub(crate) fn output_interpreter(&self) -> Result<&Interpreter, Error> {
        self.interpreters.last().ok_or(Error::GetOutputInterpreter)
    }

    fn input_interpreter(&self) -> &Interpreter {
        &self.interpreters[0]
    }

    pub(crate) fn get_input_dimensions(&self) -> Result<(usize, usize), Error> {
        let tensor = self.input_interpreter().get_input_tensor(0)?;
        Ok((tensor.dim(1)?, tensor.dim(2)?))
    }

    pub(crate) fn set_input_queue_size(&mut self, size: usize) -> Result<(), Error> {
        ffi::set_pipelined_model_runner_input_queue_size(self.runner.clone(), size)
            .map_err(Error::SetInputQueueSize)
    }

    pub(crate) fn set_output_queue_size(&mut self, size: usize) -> Result<(), Error> {
        ffi::set_pipelined_model_runner_output_queue_size(self.runner.clone(), size)
            .map_err(Error::SetOutputQueueSize)
    }

    pub(crate) fn queue_sizes(&self) -> Result<Vec<usize>, Error> {
        ffi::get_queue_sizes(&*self.runner).map_err(Error::GetQueueSizes)
    }

    #[instrument(
        name = "PipelinedModelRunner::push",
        skip(self, tensors),
        level = "debug"
    )]
    pub(crate) fn push(&self, tensors: Option<Arc<Vec<PipelineTensor>>>) -> Result<bool, Error> {
        let mut ptrs = tensors.map_or_else(Default::default, |tensors| {
            tensors
                .iter()
                .map(PipelineTensor::as_inner)
                .collect::<Vec<_>>()
        });

        ffi::push_input_tensors(self.runner.clone(), &mut ptrs).map_err(Error::PushInputTensors)
    }

    #[instrument(name = "PipelinedModelRunner::pop", skip(self), level = "debug")]
    pub(crate) fn pop(&self) -> Result<Option<Vec<UniquePtr<ffi::OutputTensor>>>, Error> {
        let mut tensors = std::iter::repeat_with(UniquePtr::null)
            .take(self.output_tensor_count)
            .collect::<Vec<_>>();
        let succeeded = ffi::pop_output_tensors(self.runner.clone(), &mut tensors)
            .map_err(Error::PopOutputTensors)?;

        Ok(if succeeded {
            debug!(message = "popped", queue_sizes = ?self.queue_sizes()?);
            Some(tensors)
        } else {
            None
        })
    }

    #[instrument(name = "PipelinedModelRunner::drain", skip(self), level = "debug")]
    fn drain(&mut self) -> Result<(), Error> {
        debug!(message = "draining");
        while self.input_queue_size()? > 0 {
            self.pop()?;
        }
        self.push(None)?;
        debug!(message = "pushed");
        while self.pop()?.is_some() {}
        debug!(message = "drained");
        assert_eq!(self.output_queue_size()?, 0);
        Ok(())
    }

    pub(crate) fn alloc_input_tensor(&self, data: &[u8]) -> Result<PipelineTensor, Error> {
        Ok(PipelineTensor::new(
            ffi::make_input_tensor(self.runner.clone(), data).map_err(Error::MakeInputTensor)?,
        ))
    }

    fn input_queue_size(&self) -> Result<usize, Error> {
        ffi::get_input_queue_size(&*self.runner).map_err(Error::GetInputQueueSize)
    }

    fn output_queue_size(&self) -> Result<usize, Error> {
        ffi::get_output_queue_size(&*self.runner).map_err(Error::GetOutputQueueSize)
    }
}

impl Drop for PipelinedModelRunner {
    #[instrument(name = "PipelinedModelRunner::drop", skip(self), level = "debug")]
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            error!(message = "failed to drain output queue", ?error);
        } else {
            debug!(message = "fully drained");
        }
    }
}

pub(crate) struct PipelineTensor(SharedPtr<ffi::PipelineTensor>);

unsafe impl Send for PipelineTensor {}
unsafe impl Sync for PipelineTensor {}

impl PipelineTensor {
    fn new(tensor: SharedPtr<ffi::PipelineTensor>) -> Self {
        Self(tensor)
    }

    fn as_inner(&self) -> SharedPtr<ffi::PipelineTensor> {
        self.0.clone()
    }
}
