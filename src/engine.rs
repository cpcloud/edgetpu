use crate::{
    coral::{PipelineTensor, PipelinedModelRunner},
    decode::Decoder,
    edgetpu::Devices,
    error::Error,
    ffi::ffi,
    pose, tflite,
};
use cxx::UniquePtr;
use std::{
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

/// An engine for doing pose estimation with edge TPU devices
pub(crate) struct Engine<D> {
    model_runner: PipelinedModelRunner,
    decoder: D,
    pub(crate) timing: Mutex<Timing>,
    frame_num: AtomicUsize,
    start_inference: Mutex<Instant>,
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct Timing {
    pub(crate) inference: Duration,
    pub(crate) decoding: Duration,
}

impl<D> Engine<D> {
    /// Construct a new Engine for pose estimation.
    ///
    /// `model_path` is a valid path to a Tensorflow Lite Flatbuffer-based model.
    /// `decoder` is a type that implements the `crate::decode::Decoder` trait.
    pub(crate) fn new<P>(
        model_paths: &[P],
        decoder: D,
        input_queue_size: usize,
        output_queue_size: usize,
        threads_per_interpreter: usize,
    ) -> Result<Self, Error>
    where
        P: AsRef<Path>,
        D: Decoder,
    {
        if model_paths.is_empty() {
            return Err(Error::ConstructPoseEngine);
        }

        let devices = Devices::new()?;

        let mut interpreters = model_paths
            .iter()
            .map(|model_path| {
                tflite::Interpreter::new(
                    std::fs::canonicalize(&model_path).map_err(move |e| {
                        Error::CanonicalizePath(e, model_path.as_ref().to_path_buf())
                    })?,
                    devices.clone(),
                    threads_per_interpreter,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        interpreters.sort_by_key(|interpreter| interpreter.model_path().to_path_buf());

        let mut model_runner = PipelinedModelRunner::new(interpreters)?;
        model_runner.set_input_queue_size(input_queue_size)?;
        model_runner.set_output_queue_size(output_queue_size)?;

        decoder.validate_output_tensor_count(
            model_runner
                .output_interpreter()?
                .get_output_tensor_count()?,
        )?;

        Ok(Self {
            model_runner,
            decoder,
            timing: Default::default(),
            frame_num: Default::default(),
            start_inference: Mutex::new(Instant::now()),
        })
    }

    pub(crate) fn get_input_dimensions(&self) -> Result<(usize, usize), Error> {
        self.model_runner.get_input_dimensions()
    }

    pub(crate) fn push(&self, input_tensor: Option<Arc<Vec<PipelineTensor>>>) -> Result<(), Error> {
        *self.start_inference.lock().unwrap() = Instant::now();
        self.model_runner.push(input_tensor)?;
        Ok(())
    }

    pub(crate) fn pop(&self) -> Result<Option<Vec<UniquePtr<ffi::OutputTensor>>>, Error> {
        let result = self.model_runner.pop()?;
        let mut timing = self.timing.lock().unwrap();
        timing.inference += self.start_inference.lock().unwrap().elapsed();
        Ok(result)
    }

    pub(crate) fn alloc_input_tensor(&self, input: &[u8]) -> Result<PipelineTensor, Error> {
        self.model_runner.alloc_input_tensor(input)
    }

    pub(crate) fn decode_poses(&self) -> Result<Box<[pose::Pose]>, Error>
    where
        D: Decoder,
    {
        let mut timing = self.timing.lock().unwrap();
        let start_decoding = Instant::now();
        let poses = self
            .decoder
            .decode_output(self.model_runner.output_interpreter()?)?;
        timing.decoding += start_decoding.elapsed();

        self.frame_num.fetch_add(1, Ordering::SeqCst);
        Ok(poses)
    }

    pub(crate) fn timing(&self) -> Timing {
        *self.timing.lock().unwrap()
    }

    pub(crate) fn frame_num(&self) -> usize {
        self.frame_num.load(Ordering::SeqCst)
    }
}
