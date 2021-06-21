use crate::{decode::Decoder, error::Error, pose, tflite};
use ndarray::ArrayView3;
use std::{
    path::Path,
    time::{Duration, Instant},
};

/// An engine for doing pose estimation with edge TPU devices
pub(crate) struct Engine<D> {
    interpreter: tflite::Interpreter,
    decoder: D,
    pub(crate) timing: Timing,
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
    pub(crate) fn new<P>(model_path: P, decoder: D) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut interpreter = tflite::Interpreter::new(model_path)?;

        decoder.validate_output_tensor_count(interpreter.get_output_tensor_count())?;
        interpreter.allocate_tensors()?;

        Ok(Self {
            interpreter,
            decoder,
            timing: Default::default(),
        })
    }

    fn infer(&mut self, input: ArrayView3<u8>) -> Result<(), Error> {
        self.interpreter
            .get_input_tensor(0)?
            .copy_from_buffer(input.as_slice().ok_or(Error::GetNDArrayAsSlice)?)?;

        // run inference
        let start_inference = Instant::now();
        self.interpreter.invoke()?;
        self.timing.inference += start_inference.elapsed();
        Ok(())
    }

    pub(crate) fn detect_poses<'a, 'b: 'a>(
        &'a mut self,
        input: ArrayView3<'b, u8>,
    ) -> Result<(Box<[pose::Pose]>, Timing), Error> {
        self.infer(input)?;
        let (rows, cols, _) = input.dim();
        Ok((
            self.decoder
                .get_decoded_arrays(&mut self.interpreter, (rows, cols))?,
            self.timing,
        ))
    }
}
