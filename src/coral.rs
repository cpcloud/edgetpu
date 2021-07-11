use crate::{error::Error, ffi::ffi, tflite::Interpreter};
use cxx::{CxxVector, SharedPtr, UniquePtr};
use std::sync::Arc;
use tracing::{debug, error, instrument};

#[derive(Clone)]
pub(crate) struct PipelinedModelRunner {
    runner: SharedPtr<ffi::PipelinedModelRunner>,
    interpreters: Arc<[Interpreter]>,
    output_tensor_count: usize,
}

unsafe impl Send for PipelinedModelRunner {}

impl PipelinedModelRunner {
    pub(crate) fn new(mut interpreters: Vec<Interpreter>) -> Result<Self, Error> {
        if interpreters.is_empty() {
            return Err(Error::ConstructPipelineModelRunnerFromInterpreters);
        }

        let output_tensor_count = interpreters
            .last()
            .ok_or(Error::GetOutputInterpreter)?
            .get_output_tensor_count()?;

        let pointers = interpreters
            .iter_mut()
            .map(|interp| interp.as_inner())
            .collect::<Vec<_>>();

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
        skip(self, tensor),
        level = "debug"
    )]
    pub(crate) fn push(&self, tensor: Option<&[u8]>) -> Result<bool, Error> {
        tensor.map_or_else(
            || {
                ffi::push_input_tensor_empty(self.runner.clone())
                    .map_err(Error::PushInputTensorEmpty)
            },
            |tensor| {
                ffi::push_input_tensor(self.runner.clone(), tensor).map_err(Error::PushInputTensor)
            },
        )
    }

    #[instrument(name = "PipelinedModelRunner::pop", skip(self), level = "debug")]
    pub(crate) fn pop(&self) -> Result<Option<UniquePtr<CxxVector<ffi::OutputTensor>>>, Error> {
        let mut succeeded = false;
        let tensors = ffi::pop_output_tensors(self.runner.clone(), &mut succeeded)
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
