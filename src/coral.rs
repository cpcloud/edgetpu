use crate::{
    error::{check_null_mut, Error},
    tflite::Interpreter,
    tflite_sys,
};
use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Arc,
};

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
    interpreters: Vec<Interpreter>,
}

impl PipelinedModelRunner {
    pub(crate) fn new(mut interpreters: Vec<Interpreter>) -> Result<Self, Error> {
        let mut pointers = interpreters
            .iter_mut()
            .map(|interp| check_null_mut(interp.as_mut_ptr()).ok_or(Error::GetInterpreter))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            runner: Arc::new(RawPipelinedModelRunner(AtomicPtr::new(
                check_null_mut(unsafe {
                    tflite_sys::CoralPipelinedModelRunnerCreate(
                        pointers.as_mut_slice().as_mut_ptr(),
                        interpreters.len(),
                    )
                })
                .ok_or(Error::CreatePipelinedModelRunner)?,
            ))),
            interpreters,
        })
    }

    pub(crate) fn output_interpreter(&mut self) -> Result<&mut Interpreter, Error> {
        Ok(self
            .interpreters
            .last_mut()
            .ok_or(Error::GetOutputInterpreter)?)
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut tflite_sys::CoralPipelinedModelRunner {
        self.runner.0.load(Ordering::SeqCst)
    }

    pub(crate) fn push(&mut self, tensors: Option<Arc<Vec<PipelineInputTensor>>>) -> bool {
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
        unsafe {
            tflite_sys::CoralPipelinedModelRunnerPush(
                self.as_mut_ptr(),
                ptrs.as_mut_slice().as_mut_ptr(),
                len,
            )
        }
    }

    pub(crate) fn pop(&mut self) -> Result<Vec<PipelineOutputTensor>, Error> {
        let num_output_tensors = self.output_interpreter()?.get_output_tensor_count()?;
        let mut tensors =
            vec![unsafe { PipelineOutputTensor::new_unchecked(self.clone()) }; num_output_tensors];
        let mut raw_tensors = tensors
            .iter_mut()
            .map(|tensor| tensor.as_mut_ptr())
            .collect::<Vec<_>>();
        let mut len = 0_usize;
        let succeeded = unsafe {
            tflite_sys::CoralPipelinedModelRunnerPop(
                self.as_mut_ptr(),
                raw_tensors.as_mut_slice().as_mut_ptr(),
                &mut len,
            )
        };
        if succeeded {
            Ok(tensors)
        } else {
            Err(Error::PopPipelinedModelOutputTensors)
        }
    }

    pub(crate) fn alloc_input_tensor(&mut self, data: &[u8]) -> Result<PipelineInputTensor, Error> {
        let size = data.len();
        Ok(PipelineInputTensor::new(
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
        ))
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
    fn new(tensor: *mut tflite_sys::CoralPipelineTensor) -> Self {
        assert!(!tensor.is_null());
        Self {
            tensor: Arc::new(RawInputTensor(AtomicPtr::new(tensor))),
        }
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
    unsafe fn new_unchecked(runner: PipelinedModelRunner) -> Self {
        Self {
            tensor: Arc::new(RawOutputTensor {
                tensor: AtomicPtr::new(std::ptr::null_mut()),
                runner,
            }),
        }
    }

    fn as_mut_ptr(&self) -> *mut tflite_sys::CoralPipelineTensor {
        self.tensor.as_mut_ptr()
    }
}
