use crate::{
    error::Error,
    pose::{self, Keypoint, KeypointKind, Pose},
    tflite,
};
use ndarray::ShapeBuilder;
use num_traits::cast::FromPrimitive;
use opencv::{core::Mat, prelude::*};
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub(crate) struct Engine {
    interpreter: tflite::Interpreter,
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Timing {
    copy: Duration,
    inference: Duration,
    post_proc: Duration,
}

impl Engine {
    pub(crate) fn new<P>(model_path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let mut interpreter = tflite::Interpreter::new(model_path)?;
        interpreter.allocate_tensors()?;
        Ok(Self { interpreter })
    }

    pub(crate) fn detect_poses(&mut self, input: &Mat) -> Result<(Vec<pose::Pose>, Timing), Error> {
        let bytes = input.data_typed::<u8>().map_err(Error::GetTypedData)?;
        let sizes = self.interpreter.get_output_tensor_count();
        assert_eq!(sizes, 4);

        let start_copy = Instant::now();
        let mut input_tensor = self.interpreter.get_input_tensor(0)?;
        input_tensor.copy_from_buffer(bytes)?;
        let copy = start_copy.elapsed();

        let start_inference = Instant::now();
        self.interpreter.invoke()?;
        let inference = start_inference.elapsed();

        let start_post_proc = Instant::now();

        let pose_keypoints = self.interpreter.get_output_tensor(0)?;
        let pose_keypoints = pose_keypoints.as_ndarray(
            *(
                pose_keypoints.dim(0),
                pose_keypoints.dim(1),
                pose_keypoints.dim(2),
            )
                .into_shape()
                .raw_dim(),
        )?;
        let keypoint_scores = self.interpreter.get_output_tensor(1)?;
        let keypoint_scores = keypoint_scores.as_ndarray(
            *(keypoint_scores.dim(0), keypoint_scores.dim(1))
                .into_shape()
                .raw_dim(),
        )?;
        let pose_scores = self.interpreter.get_output_tensor(2)?;
        let pose_scores = pose_scores.as_ndarray(*pose_scores.dim(0).into_shape().raw_dim())?;
        let nposes = self.interpreter.get_output_tensor(3)?.as_slice()[0] as usize;

        let mut poses = Vec::with_capacity(nposes);

        for (pose_i, keypoint) in pose_keypoints.axis_iter(ndarray::Axis(0)).enumerate() {
            let mut keypoint_map: pose::Keypoints = Default::default();

            for (point_i, point) in keypoint.axis_iter(ndarray::Axis(1)).enumerate() {
                let keypoint = Keypoint {
                    kind: Some(
                        KeypointKind::from_usize(point_i)
                            .ok_or(Error::ConvertUSizeToKeypointKind(point_i))?,
                    ),
                    point: opencv::core::Point2f::new(point[1], point[0]),
                    score: keypoint_scores[(pose_i, point_i)],
                };

                keypoint_map[point_i] = keypoint;
            }

            poses.push(Pose::new(keypoint_map, pose_scores[pose_i]));
        }
        let post_proc = start_post_proc.elapsed();

        Ok((
            poses,
            Timing {
                copy,
                inference,
                post_proc,
            },
        ))
    }
}
