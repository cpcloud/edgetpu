use crate::{
    coral::{PipelineInputTensor, PipelineOutputTensor, PipelinedModelRunner},
    decode::Decoder,
    error::Error,
    pose, tflite,
};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{sync_channel, Receiver, SyncSender},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

/// An engine for doing pose estimation with edge TPU devices
pub(crate) struct Engine<D> {
    model_runner: PipelinedModelRunner,
    decoder: D,
    pub(crate) timing: Timing,
    output_tensors_rx: Receiver<Vec<PipelineOutputTensor>>,
    input_tensors_tx: SyncSender<Arc<Vec<PipelineInputTensor>>>,
    _producer: JoinHandle<Result<(), Error>>,
    _consumer: JoinHandle<Result<(), Error>>,
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct Timing {
    pub(crate) inference: Duration,
}

impl<D> Engine<D>
where
    D: Decoder,
{
    /// Construct a new Engine for pose estimation.
    ///
    /// `model_path` is a valid path to a Tensorflow Lite Flatbuffer-based model.
    /// `decoder` is a type that implements the `crate::decode::Decoder` trait.
    pub(crate) fn new<P>(
        model_paths: &[P],
        decoder: D,
        running: Arc<AtomicBool>,
    ) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        if model_paths.is_empty() {
            return Err(Error::ConstructPoseEngine);
        }

        let interpreters = model_paths
            .iter()
            .map(|model_path| {
                Ok(tflite::Interpreter::new(
                    std::fs::canonicalize(model_path).map_err(|e| {
                        Error::CanonicalizePath(e, model_path.as_ref().to_path_buf())
                    })?,
                )?)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut model_runner = PipelinedModelRunner::new(interpreters)?;
        let output_interpreter = model_runner.output_interpreter()?;
        decoder.validate_output_tensor_count(output_interpreter.get_output_tensor_count()?)?;

        let mut input_runner = model_runner.clone();

        let (output_tensors_tx, output_tensors_rx) = sync_channel(100);
        let (input_tensors_tx, input_tensors_rx) = sync_channel(100);
        let producer = std::thread::spawn(move || loop {
            if !running.load(Ordering::SeqCst) {
                input_runner.push(None);
                return Ok(());
            }

            input_runner.push(Some(input_tensors_rx.recv().unwrap()));
        });

        // channel: std::sync::mpsc::Sender<Vec<*mut tflite_sys::CoralPipelineTensor>>,
        let mut output_runner = model_runner.clone();
        let consumer = std::thread::spawn(move || loop {
            output_tensors_tx.send(output_runner.pop()?).unwrap();
        });

        Ok(Self {
            model_runner,
            decoder,
            timing: Default::default(),
            output_tensors_rx,
            input_tensors_tx,
            _producer: producer,
            _consumer: consumer,
        })
    }

    fn infer<'a, 'b: 'a>(
        &'a mut self,
        input: &'b [u8],
    ) -> Result<Vec<PipelineOutputTensor>, Error> {
        let input_tensor = self.model_runner.alloc_input_tensor(input)?;
        let inputs = Arc::new(vec![input_tensor]);
        let _guard = inputs.clone();
        self.input_tensors_tx.send(inputs).unwrap();
        Ok(self.output_tensors_rx.recv().unwrap())
    }

    pub(crate) fn detect_poses<'a, 'b: 'a>(
        &'a mut self,
        input: &'b [u8],
    ) -> Result<(Box<[pose::Pose]>, Timing), Error> {
        let start_inference = Instant::now();
        let _output_tensors = self.infer(input)?;
        self.timing.inference += start_inference.elapsed();

        let result = self
            .decoder
            .decode_output(self.model_runner.output_interpreter()?)?;
        Ok((result, self.timing))
    }
}
