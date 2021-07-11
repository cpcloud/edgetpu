use crate::{
    coral::PipelinedModelRunner, decode::Decoder, edgetpu::Devices, error::Error, ffi::ffi, pose,
    tflite,
};
use cxx::{CxxVector, UniquePtr};
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing::{debug, instrument};

/// An engine for doing pose estimation with edge TPU devices
#[derive(Clone)]
pub(crate) struct Engine<D> {
    model_runner: Arc<Mutex<PipelinedModelRunner>>,
    decoder: Arc<D>,
    pub(crate) timing: Arc<Mutex<Timing>>,
    start_inference: Arc<Mutex<Instant>>,
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
                let path = model_path.as_ref();
                tflite::Interpreter::new(
                    path.canonicalize()
                        .map_err(move |e| Error::CanonicalizePath(e, path.to_path_buf()))?,
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
            model_runner: Arc::new(Mutex::new(model_runner)),
            decoder: Arc::new(decoder),
            timing: Default::default(),
            start_inference: Arc::new(Mutex::new(Instant::now())),
        })
    }

    pub(crate) fn get_input_dimensions(&self) -> Result<(usize, usize), Error> {
        self.model_runner.lock().unwrap().get_input_dimensions()
    }

    #[instrument(name = "Engine::push", skip(self, input_tensor), level = "debug")]
    pub(crate) fn push(&self, input_tensor: Option<&[u8]>) -> Result<(), Error> {
        *self.start_inference.lock().unwrap() = Instant::now();
        self.model_runner.lock().unwrap().push(input_tensor)?;
        debug!(duration = ?self.start_inference.lock().unwrap().elapsed());
        Ok(())
    }

    #[instrument(name = "Engine::pop", skip(self), level = "debug")]
    pub(crate) fn pop(&self) -> Result<Option<UniquePtr<CxxVector<ffi::OutputTensor>>>, Error> {
        let result = self.model_runner.lock().unwrap().pop()?;
        let mut timing = self.timing.lock().unwrap();
        let elapsed = self.start_inference.lock().unwrap().elapsed();
        timing.inference += elapsed;
        Ok(result)
    }

    pub(crate) fn decode_poses(&self) -> Result<Box<[pose::Pose]>, Error>
    where
        D: Decoder,
    {
        let mut timing = self.timing.lock().unwrap();
        let start_decoding = Instant::now();
        let poses = self
            .decoder
            .decode_output(self.model_runner.lock().unwrap().output_interpreter()?)?;
        timing.decoding += start_decoding.elapsed();

        Ok(poses)
    }

    pub(crate) fn timing(&self) -> Timing {
        *self.timing.lock().unwrap()
    }
}
