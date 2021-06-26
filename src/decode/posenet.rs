use crate::{error::Error, pose, tflite};
use std::ops::DerefMut;

#[derive(Debug, Clone, Copy, Default, structopt::StructOpt)]
pub(crate) struct Decoder {}

impl crate::decode::Decoder for Decoder {
    fn expected_output_tensors(&self) -> usize {
        4
    }

    fn decode_output<I>(&self, interp: I) -> Result<Box<[pose::Pose]>, Error>
    where
        I: DerefMut<Target = tflite::Interpreter>,
    {
        // construct the output tensors
        let pose_keypoints = interp.get_output_tensor_by_name("poses")?;
        let pose_keypoints = pose_keypoints.as_ndarray(
            pose_keypoints.as_f32_slice()?,
            (
                pose_keypoints.dim(1)?,
                pose_keypoints.dim(2)?,
                pose_keypoints.dim(3)?,
            ),
        )?;
        let keypoint_scores = interp.get_output_tensor_by_name("poses:1")?;
        let keypoint_scores = keypoint_scores.as_ndarray(
            keypoint_scores.as_f32_slice()?,
            (keypoint_scores.dim(1)?, keypoint_scores.dim(2)?),
        )?;
        let pose_scores = interp.get_output_tensor(2)?;
        let pose_scores =
            pose_scores.as_ndarray(pose_scores.as_f32_slice()?, pose_scores.dim(1)?)?;
        crate::decode::reconstruct_from_arrays(
            pose_keypoints.into(),
            keypoint_scores.into(),
            pose_scores.into(),
        )
    }
}
