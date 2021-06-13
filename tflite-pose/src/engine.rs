use crate::{errors::Error, pose, tflite};
use opencv::{core::Mat, prelude::*};
use std::{path::Path, time::Duration};

pub(crate) struct Engine {
    interpreter: tflite::Interpreter,
    input_tensor_shape: Vec<usize>,
    output_offsets: Vec<usize>,
    mirror: bool,
}

const NANOS_PER_MILLI: f64 = 1_000_000.0;

impl Engine {
    pub(crate) fn new<P>(model_path: P, mirror: bool) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let interpreter = tflite::Interpreter::new(model_path)?;
        let input_tensor_shape = interpreter
            .get_input_tensor_shape()
            .into_iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        let output_offsets = interpreter
            .get_all_output_tensors_sizes()
            .iter()
            .copied()
            .scan(0, |result, shape| {
                *result += shape;
                Some(*result)
            })
            .collect();
        Ok(Self {
            interpreter,
            input_tensor_shape,
            output_offsets,
            mirror,
        })
    }

    fn width(&self) -> usize {
        self.input_tensor_shape[2]
    }

    fn depth(&self) -> usize {
        self.input_tensor_shape[3]
    }

    fn detect_poses(&mut self, input: &Mat) -> Result<(Vec<pose::Pose>, Duration), Error> {
        let bytes = input
            .data_typed::<u8>()
            .map_err(Error::GetTypedData)?
            .to_owned();

        let sizes = self.interpreter.get_all_output_tensors_sizes();
        let mut keypoints = vec![0; sizes[0]];
        let mut keypoint_scores = vec![0; sizes[1]];
        let mut pose_scores = vec![0; sizes[2]];
        let nposes = self.interpreter.run_inference(bytes);
        let inference_time = Duration::from_nanos(
            (f64::from(self.interpreter.get_inference_time()) * NANOS_PER_MILLI) as u64,
        );

        Ok((vec![], inference_time))
    }
}
