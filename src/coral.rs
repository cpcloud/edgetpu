use std::{
    ffi::c_void,
    marker::PhantomData,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicPtr, Ordering},
        Arc,
    },
    thread::JoinHandle,
};

use crate::{error::Error, tflite::Tensor, tflite_sys};

// #[derive(Debug)]
// pub(crate) struct PipelineTensor {
//     tensor: *mut tflite_sys::CoralPipelineTensor,
// }
//
// impl PipelineTensor {
//     fn new(tensor: *mut tflite_sys::CoralPipelineTensor) -> Self {
//         Self { tensor }
//     }
// }
//
// #[derive(Clone)]
// pub(crate) struct PipelinedModelRunner {
//     runner: Arc<AtomicPtr<*mut tflite_sys::CoralPipelinedModelRunner>>,
//     allocator: DefaultAllocator,
// }
//
// impl PipelinedModelRunner {
//     pub(crate) fn new(model_segments: Vec<PathBuf>) -> Result<Self, Error> {
//         let runner = unsafe { tflite_sys::CoralPipelinedModelRunnerCreate(model_segments) };
//         Ok(Self {
//             runner: Arc::new(AtomicPtr::new(runner)),
//             allocator: Arc::new(DefaultAllocator::new()),
//         })
//     }
//
//     pub(crate) fn producer(&mut self, stop: Arc<AtomicBool>) -> JoinHandle<Result<(), Error>> {
//         let runner = self.clone();
//         std::thread::spawn(move || loop {
//             if stop.load(Ordering::SeqCst) {
//                 runner.push(None);
//                 return Ok(());
//             }
//
//             runner.push(Some(self.allocator.create_input_tensors()));
//         })
//     }
//
//     pub(crate) fn consumer(
//         &mut self,
//         channel: std::sync::mpsc::Sender<Vec<PipelineTensor>>,
//     ) -> JoinHandle<Result<(), Error>> {
//         let runner = self.clone();
//         std::thread::spawn(move || {
//             while let Some(output) = runner.pop() {
//                 channel
//                     .send(output)
//                     .map_err(Error::SendPipelinedOutputTensors)?;
//             }
//             Ok(())
//         })
//     }
//
//     pub(crate) fn push(&mut self, tensors: Option<Vec<tflite_sys::CoralPipelineTensor>>) {
//         let (ptr, len) = tensors.map_or_else(
//             || (std::ptr::null(), 0),
//             |tensors| (tensors.as_ptr(), tensors.len()),
//         );
//         unsafe {
//             tflite_sys::CoralPipelinedModelRunnerPush(self.runner.load(Ordering::SeqCst), ptr, len)
//         }
//     }
//
//     pub(crate) fn pop(&mut self) -> Option<Vec<PipelineTensor>> {
//         let (ptr, len) =
//             unsafe { tflite_sys::CoralPipelinedModelRunnerPop(self.runner.load(Ordering::SeqCst)) };
//         (0..len)
//             .map(|offset| PipelineTensor::new(unsafe { ptr.add(offset) }))
//             .collect()
//     }
// }
//
// impl Drop for PipelinedModelRunner {
//     fn drop(&mut self) {
//         unsafe {
//             tflite_sys::CoralPipelinedModelRunnerDestroy(self.runner.load(Ordering::SeqCst));
//         }
//     }
// }

struct DefaultAllocator {
    allocator: *mut c_void,
}

impl DefaultAllocator {
    fn new() -> Self {
        Self {
            allocator: unsafe { tflite_sys::CoralDefaultAllocatorCreate() },
        }
    }
}

impl Drop for DefaultAllocator {
    fn drop(&mut self) {
        unsafe {
            tflite_sys::CoralDefaultAllocatorDestroy(self.allocator);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DefaultAllocator;

    #[test]
    fn construct_allocator() {
        let _alloc = DefaultAllocator::new();
    }
}
