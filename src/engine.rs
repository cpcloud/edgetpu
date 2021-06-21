use crate::{
    error::Error,
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
        assert_eq!(interpreter.get_output_tensor_count(), 4);

        interpreter.allocate_tensors()?;
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

        // copy the bytes into the input tensor
        self.interpreter
            .get_input_tensor(0)?
            .copy_from_raw_buffer(raw_data, number_of_elements)?;

        // run inference
        let start_inference = Instant::now();
        self.interpreter.invoke()?;
        self.timing.inference += start_inference.elapsed();

        // construct the output tensors
        let pose_keypoints = self.interpreter.get_output_tensor(0)?;
        let pose_keypoints = pose_keypoints.as_ndarray((
            pose_keypoints.dim(1),
            pose_keypoints.dim(2),
            pose_keypoints.dim(3),
        ))?;

        let keypoint_scores = self.interpreter.get_output_tensor(1)?;
        let keypoint_scores =
            keypoint_scores.as_ndarray((keypoint_scores.dim(1), keypoint_scores.dim(2)))?;

        let pose_scores = self.interpreter.get_output_tensor(2)?;
        let pose_scores = pose_scores.as_ndarray(pose_scores.dim(1))?;

        let nposes = self.interpreter.get_output_tensor(3)?.as_slice()[0]
            .to_usize()
            .ok_or(Error::ConvertNumPosesToUSize)?;

        // move poses into a more useful structure
        let mut poses = Vec::with_capacity(nposes);

        for (pose_i, (keypoint, &pose_score)) in pose_keypoints
            .axis_iter(Axis(0))
            .zip(pose_scores)
            .enumerate()
        {
            let mut keypoints: pose::Keypoints = Default::default();

            for (point_i, point) in keypoint.axis_iter(Axis(0)).enumerate() {
                keypoints[point_i] = Keypoint {
                    kind: Some(
                        KeypointKind::from_usize(point_i)
                            .ok_or(Error::ConvertUSizeToKeypointKind(point_i))?,
                    ),
                    point: opencv::core::Point2f::new(point[1], point[0]),
                    score: keypoint_scores[(pose_i, point_i)],
                };
            }

            poses.push(Pose {
                keypoints,
                score: pose_score,
            });
        }

        Ok(poses)
    }
}
