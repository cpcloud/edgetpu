use crate::{
    error::{check_null_mut, Error},
    tflite::Interpreter,
    tflite_sys,
};
use more_asserts::assert_gt;
use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Arc,
};
use tracing::{debug, instrument};

struct RawPipelinedModelRunner(AtomicPtr<tflite_sys::CoralPipelinedModelRunner>);

impl Drop for RawPipelinedModelRunner {
    fn drop(&mut self) {
        unsafe {
            tflite_sys::CoralPipelinedModelRunnerDestroy(self.0.load(Ordering::SeqCst));
        }
    }
}

#[derive(Clone)]
pub(crate) struct PipelinedModelRunner {
    runner: Arc<RawPipelinedModelRunner>,
    interpreters: Box<[Interpreter]>,
}

impl PipelinedModelRunner {
    pub(crate) fn new(mut interpreters: Vec<Interpreter>) -> Result<Self, Error> {
        if interpreters.is_empty() {
            return Err(Error::ConstructPipelineModelRunnerFromInterpreters);
        }

        let mut pointers = interpreters
            .iter_mut()
            .map(|interp| check_null_mut(interp.as_mut_ptr()).ok_or(Error::GetInterpreter))
            .collect::<Result<Vec<_>, _>>()?;

        if pointers.is_empty() {
            return Err(Error::GetInterpreterPointers);
        }

        Ok(Self {
            runner: Arc::new(RawPipelinedModelRunner(AtomicPtr::new(
                check_null_mut(unsafe {
                    tflite_sys::CoralPipelinedModelRunnerCreate(
                        check_null_mut(pointers.as_mut_slice().as_mut_ptr())
                            .ok_or(Error::GetInterpreterVecPointer)?,
                        interpreters.len(),
                    )
                })
                .ok_or(Error::CreatePipelinedModelRunner)?,
            ))),
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
        unsafe { tflite_sys::CoralPipelinedModelRunnerSetInputQueueSize(self.as_mut_ptr(), size) }
    }

    pub(crate) fn set_output_queue_size(&mut self, size: usize) {
        unsafe { tflite_sys::CoralPipelinedModelRunnerSetOutputQueueSize(self.as_mut_ptr(), size) }
    }

    pub(crate) fn segment_stats(&self) -> Result<Vec<tflite_sys::CoralSegmentStats>, Error> {
        let mut n = 0_usize;
        let stats = check_null_mut(unsafe {
            tflite_sys::CoralPipelinedModelRunnerGetSegmentStats(self.as_mut_ptr(), &mut n)
        })
        .ok_or(Error::GetSegmentStats)?;

        if n == 0 {
            return Err(Error::EmptySegmentStats);
        }

        let mut result = Vec::with_capacity(n);
        unsafe {
            std::ptr::copy(stats, result.as_mut_ptr(), n);
            result.set_len(n);
            tflite_sys::CoralPipelinedModelRunnerDestroySegmentStats(stats);
        }
        Ok(result)
    }

    pub(crate) fn as_mut_ptr(&self) -> *mut tflite_sys::CoralPipelinedModelRunner {
        self.runner.0.load(Ordering::SeqCst)
    }

    pub(crate) fn queue_sizes(&self) -> Result<Vec<usize>, Error> {
        let mut n = 0_usize;
        let sizes = check_null_mut(unsafe {
            tflite_sys::CoralPipelinedModelRunnerGetQueueSizes(self.as_mut_ptr(), &mut n)
        })
        .ok_or(Error::GetQueueSizesPointer)?;

        let mut result = Vec::with_capacity(n);
        unsafe {
            std::ptr::copy(sizes, result.as_mut_ptr(), n);
            result.set_len(n);
            tflite_sys::CoralPipelinedModelRunnerDestroyQueueSizes(sizes);
        }
        Ok(result)
    }

    #[instrument(
        name = "PipelinedModelRunner::push",
        skip(self, tensors),
        level = "debug"
    )]
    pub(crate) fn push(&self, tensors: Option<Arc<Vec<PipelineInputTensor>>>) -> bool {
        let (mut ptrs, len) = tensors.map_or_else(
            || (vec![], 0),
            |tensors| {
                let len = tensors.len();
                (
                    tensors
                        .iter()
                        .map(|tensor| tensor.as_mut_ptr())
                        .collect::<Vec<_>>(),
                    len,
                )
            },
        );
        debug!("pushing to input queue {:?}", self.queue_sizes().unwrap());
        unsafe {
            tflite_sys::CoralPipelinedModelRunnerPush(
                self.as_mut_ptr(),
                ptrs.as_mut_slice().as_mut_ptr(),
                len,
            )
        }
    }

    #[instrument(name = "PipelinedModelRunner::pop", skip(self), level = "debug")]
    pub(crate) fn pop(&self) -> Result<Vec<PipelineOutputTensor>, Error> {
        let num_output_tensors = self.output_interpreter()?.get_output_tensor_count()?;
        let mut tensors = vec![std::ptr::null_mut(); num_output_tensors];
        let mut len = 0_usize;
        let succeeded = unsafe {
            tflite_sys::CoralPipelinedModelRunnerPop(
                self.as_mut_ptr(),
                tensors.as_mut_slice().as_mut_ptr(),
                &mut len,
            )
        };

        assert_gt!(len, 0, "got zero output tensors");

        if succeeded {
            debug!("popping from output queue {:?}", self.queue_sizes()?);
            tensors
                .into_iter()
                .map(|ptr| PipelineOutputTensor::new(ptr, self.clone()))
                .collect::<Result<Vec<_>, _>>()
        } else {
            Err(Error::PopPipelinedModelOutputTensors)
        }
    }

    pub(crate) fn alloc_input_tensor(&self, data: &[u8]) -> Result<PipelineInputTensor, Error> {
        let size = data.len();
        PipelineInputTensor::new(
            check_null_mut(unsafe {
                let raw = tflite_sys::CoralPipelineTensorCreate(
                    self.as_mut_ptr(),
                    size,
                    tflite_sys::TfLiteType::kTfLiteUInt8,
                );
                tflite_sys::CoralPipelineTensorCopyFromBuffer(raw, data.as_ptr().cast(), size);
                raw
            })
            .ok_or(Error::AllocInputTensor)?,
        )
    }
}

struct RawInputTensor(AtomicPtr<tflite_sys::CoralPipelineTensor>);

impl RawInputTensor {
    fn as_mut_ptr(&self) -> *mut tflite_sys::CoralPipelineTensor {
        self.0.load(Ordering::SeqCst)
    }
}

impl Drop for RawInputTensor {
    fn drop(&mut self) {
        unsafe {
            tflite_sys::CoralPipelineInputTensorDestroy(self.as_mut_ptr());
        }
    }
}

/// The buffer inside these are owned by the allocator, so we do not implement Drop.
#[derive(Clone)]
pub(crate) struct PipelineInputTensor {
    tensor: Arc<RawInputTensor>,
}

impl PipelineInputTensor {
    fn new(tensor: *mut tflite_sys::CoralPipelineTensor) -> Result<Self, Error> {
        Ok(Self {
            tensor: Arc::new(RawInputTensor(AtomicPtr::new(
                check_null_mut(tensor).ok_or(Error::GetPipelineInputTensor)?,
            ))),
        })
    }

    fn as_mut_ptr(&self) -> *mut tflite_sys::CoralPipelineTensor {
        self.tensor.as_mut_ptr()
    }
}

struct RawOutputTensor {
    tensor: AtomicPtr<tflite_sys::CoralPipelineTensor>,
    runner: PipelinedModelRunner,
}

impl RawOutputTensor {
    fn as_mut_ptr(&self) -> *mut tflite_sys::CoralPipelineTensor {
        self.tensor.load(Ordering::SeqCst)
    }
}

impl Drop for RawOutputTensor {
    fn drop(&mut self) {
        unsafe {
            tflite_sys::CoralPipelineOutputTensorDestroy(
                self.runner.as_mut_ptr(),
                self.as_mut_ptr(),
            );
        }
    }
}

/// The buffer inside these are NOT owned by the allocator, so we implement Drop.
#[derive(Clone)]
pub(crate) struct PipelineOutputTensor {
    tensor: Arc<RawOutputTensor>,
}

impl PipelineOutputTensor {
    fn new(
        tensor: *mut tflite_sys::CoralPipelineTensor,
        runner: PipelinedModelRunner,
    ) -> Result<Self, Error> {
        Ok(Self {
            tensor: Arc::new(RawOutputTensor {
                tensor: AtomicPtr::new(
                    check_null_mut(tensor).ok_or(Error::GetPipelineOutputTensor)?,
                ),
                runner,
            }),
        })
    }
}
