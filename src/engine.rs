use crate::{
    error::Error,
    pose::{self, Keypoint, KeypointKind, Pose},
    tflite,
};
use ndarray::ShapeBuilder;
use num_traits::cast::{FromPrimitive, ToPrimitive};
use opencv::{core::Mat, prelude::*};
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub(crate) struct Engine {
    interpreter: tflite::Interpreter,
    pub(crate) timing: Timing,
}

#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct Timing {
    pub(crate) inference: Duration,
}

impl Engine {
    pub(crate) fn new<P>(model_path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut interpreter = tflite::Interpreter::new(model_path)?;
        interpreter.allocate_tensors()?;
        assert_eq!(interpreter.get_output_tensor_count(), 4);
        Ok(Self {
            interpreter,
            timing: Default::default(),
        })
    }

    pub(crate) fn detect_poses(&mut self, input: &Mat) -> Result<Vec<pose::Pose>, Error> {
        let raw_data = input.data().map_err(Error::GetMatData)? as _;
        let number_of_elements = input.total().map_err(Error::GetTotalNumberOfElements)?
            * input
                .channels()
                .map_err(Error::GetChannels)?
                .to_usize()
                .unwrap();
        // SAFETY: raw_data is a valid pointer, points to data of length `number_of_elements`
        // and lives <= the lifetime of `input`
        let bytes = unsafe { std::slice::from_raw_parts(raw_data, number_of_elements) };

        // copy the bytes into the input tensor
        self.interpreter
            .get_input_tensor(0)?
            .copy_from_buffer(bytes)?;

        // run inference
        let start_inference = Instant::now();
        self.interpreter.invoke()?;
        self.timing.inference += start_inference.elapsed();

        // construct the output tensors
        let pose_keypoints = self.interpreter.get_output_tensor(0)?;
        let pose_keypoints = pose_keypoints.as_ndarray(
            *(
                pose_keypoints.dim(1),
                pose_keypoints.dim(2),
                pose_keypoints.dim(3),
            )
                .into_shape()
                .raw_dim(),
        )?;
        let keypoint_scores = self.interpreter.get_output_tensor(1)?;
        let keypoint_scores = keypoint_scores.as_ndarray(
            *(keypoint_scores.dim(1), keypoint_scores.dim(2))
                .into_shape()
                .raw_dim(),
        )?;
        let pose_scores = self.interpreter.get_output_tensor(2)?;
        let pose_scores = pose_scores.as_ndarray(*pose_scores.dim(1).into_shape().raw_dim())?;
        let nposes = self.interpreter.get_output_tensor(3)?.as_slice()[0] as usize;

        // move poses into a more useful structure
        let mut poses = Vec::with_capacity(nposes);

        for (pose_i, keypoint) in pose_keypoints.axis_iter(ndarray::Axis(0)).enumerate() {
            let mut keypoint_map: pose::Keypoints = Default::default();

            for (point_i, point) in keypoint.axis_iter(ndarray::Axis(0)).enumerate() {
                keypoint_map[point_i] = Keypoint {
                    kind: Some(
                        KeypointKind::from_usize(point_i)
                            .ok_or(Error::ConvertUSizeToKeypointKind(point_i))?,
                    ),
                    point: opencv::core::Point2f::new(point[1], point[0]),
                    score: keypoint_scores[(pose_i, point_i)],
                };
            }

            poses.push(Pose {
                keypoints: keypoint_map,
                score: pose_scores[pose_i],
            });
        }

        Ok(poses)
    }
}
