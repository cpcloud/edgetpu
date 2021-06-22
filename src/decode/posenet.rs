use crate::{error::Error, pose, tflite};

#[derive(Debug, Clone, Copy, Default, structopt::StructOpt)]
pub(crate) struct Decoder {}

impl crate::decode::Decoder for Decoder {
    fn expected_output_tensors(&self) -> usize {
        4
    }

    fn get_decoded_arrays<'a, 'b: 'a>(
        &'a self,
        interp: &'b mut tflite::Interpreter,
        (_width, _height): (usize, usize),
    ) -> Result<Box<[pose::Pose]>, Error> {
        // construct the output tensors
        let pose_keypoints = interp.get_output_tensor(0)?;
        let pose_keypoints = pose_keypoints.as_ndarray(
            unsafe { pose_keypoints.as_slice::<f32>() },
            (
                pose_keypoints.dim(1)?,
                pose_keypoints.dim(2)?,
                pose_keypoints.dim(3)?,
            ),
        )?;
        let keypoint_scores = interp.get_output_tensor(1)?;
        let keypoint_scores = keypoint_scores.as_ndarray(
            unsafe { keypoint_scores.as_slice::<f32>() },
            (keypoint_scores.dim(1)?, keypoint_scores.dim(2)?),
        )?;
        let pose_scores = interp.get_output_tensor(2)?;
        let pose_scores = pose_scores.as_ndarray(
            unsafe { pose_scores.as_slice::<f32>() },
            pose_scores.dim(1)?,
        )?;
        crate::decode::reconstruct_from_arrays(
            pose_keypoints.into(),
            keypoint_scores.into(),
            pose_scores.into(),
        )
    }
}
