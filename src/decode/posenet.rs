use crate::{
    error::Error,
    pose::{self, Keypoint, KeypointKind, Pose},
    tflite,
};
use num_traits::cast::{FromPrimitive, ToPrimitive};

#[derive(Debug, Clone, Copy, Default, structopt::StructOpt)]
pub(crate) struct Decoder {}

impl crate::decode::Decoder for Decoder {
    fn expected_output_tensors(&self) -> usize {
        4
    }

    fn decode(
        &self,
        interp: &mut tflite::Interpreter,
        (_width, _height): (u16, u16),
    ) -> Result<Vec<pose::Pose>, Error> {
        // construct the output tensors
        let pose_keypoints = interp.get_output_tensor(0)?;
        let pose_keypoints = pose_keypoints.as_ndarray((
            pose_keypoints.dim(1)?,
            pose_keypoints.dim(2)?,
            pose_keypoints.dim(3)?,
        ))?;
        let keypoint_scores = interp.get_output_tensor(1)?;
        let keypoint_scores =
            keypoint_scores.as_ndarray((keypoint_scores.dim(1)?, keypoint_scores.dim(2)?))?;
        let pose_scores = interp.get_output_tensor(2)?;
        let pose_scores = pose_scores.as_ndarray(pose_scores.dim(1)?)?;
        let nposes = interp.get_output_tensor(3)?.as_slice()[0]
            .to_usize()
            .ok_or(Error::NumPosesToUSize)?;

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
