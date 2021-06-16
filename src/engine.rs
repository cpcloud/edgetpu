use crate::{
    decode::Decoder,
    error::Error,
    pose,
    pose::{self, Keypoint, KeypointKind, Pose},
    tflite,
};
use ndarray::Axis;
use num_traits::cast::{FromPrimitive, ToPrimitive};
use opencv::{core::Mat, prelude::*};
use std::{
    path::Path,
    time::{Duration, Instant},
};

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
    pub(crate) fn new<P>(model_path: P, decoder: D) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut interpreter = tflite::Interpreter::new(model_path)?;

        D::validate_output_tensor_count(interpreter.get_output_tensor_count()?)?;
        interpreter.allocate_tensors()?;

        Ok(Self {
            interpreter,
            decoder,
            timing: Default::default(),
        })
    }

    fn infer(&mut self, input: &Mat) -> Result<(), Error> {
        let step = input.step1(0).map_err(Error::GetChannels)?
            * input.elem_size1().map_err(Error::GetElemSize1)?;
        let rows = input.rows().to_usize().ok_or(Error::ConvertRowsToUsize)?;
        let num_elements = step * rows;

        // SAFETY: raw_data is a valid pointer, points to data of length `number_of_elements`
        // and lives <= the lifetime of `input`
        let raw_data = input.data().map_err(Error::GetMatData)? as _;
        let bytes = unsafe { std::slice::from_raw_parts(raw_data, num_elements) };

        // copy the bytes into the input tensor
        self.interpreter
            .get_input_tensor(0)?
            .copy_from_raw_buffer(raw_data, number_of_elements)?;

        // run inference
        let start_inference = Instant::now();
        self.interpreter.invoke()?;
        self.timing.inference += start_inference.elapsed();
        Ok(())
    }

    pub(crate) fn detect_poses(&mut self, input: &Mat) -> Result<Vec<pose::Pose>, Error> {
        self.infer(input)?;
        self.decoder.decode(&mut self.interpreter)
    }
}
