use crate::{error::Error, pose, tflite};
use ndarray::{Axis, CowArray, Ix1, Ix2, Ix3};
use num_traits::cast::FromPrimitive;

/// Decode poses into a Vec of Pose.
pub(self) fn reconstruct_from_arrays(
    pose_keypoints: CowArray<f32, Ix3>,
    keypoint_scores: CowArray<f32, Ix2>,
    pose_scores: CowArray<f32, Ix1>,
) -> Result<Box<[pose::Pose]>, Error> {
    eprintln!("{:#?}", pose_keypoints);
    pose_scores
        .indexed_iter()
        .zip(pose_keypoints.axis_iter(Axis(0)))
        .map(move |((pose_i, &score), keypoint)| {
            let mut keypoints: pose::Keypoints = Default::default();

            for (point_i, point) in keypoint.axis_iter(Axis(0)).enumerate() {
                keypoints[point_i] = pose::Keypoint {
                    kind: Some(
                        pose::KeypointKind::from_usize(point_i)
                            .ok_or(Error::ConvertUSizeToKeypointKind(point_i))?,
                    ),
                    point: opencv::core::Point2f::new(point[1], point[0]),
                    score: keypoint_scores[(pose_i, point_i)],
                };
            }

            Ok(pose::Pose { keypoints, score })
        })
        .collect::<Result<Box<_>, _>>()
}

pub(crate) trait Decoder {
    /// Return the number of expected_output_tensors the decoder expects to operate on.
    fn expected_output_tensors(&self) -> usize;

    fn get_decoded_arrays(
        &self,
        interp: &mut tflite::Interpreter,
    ) -> Result<Box<[pose::Pose]>, Error>;

    /// Validate that the model has the expected number of output tensors.
    fn validate_output_tensor_count(&self, output_tensor_count: usize) -> Result<(), Error> {
        let expected_output_tensors = self.expected_output_tensors();
        if output_tensor_count != expected_output_tensors {
            Err(Error::GetExpectedNumOutputs(
                expected_output_tensors,
                output_tensor_count,
            ))
        } else {
            Ok(())
        }
    }
}

mod hand_rolled;
#[cfg(feature = "posenet_decoder")]
mod posenet;

#[derive(Debug, structopt::StructOpt)]
pub(crate) enum Decode {
    /// Decode using the builtin PosenetDecoderOp
    #[cfg(feature = "posenet_decoder")]
    Posenet(posenet::Decoder),
    /// Decode using a hand rolled decoder
    HandRolled(hand_rolled::Decoder),
}

impl Default for Decode {
    fn default() -> Self {
        #[cfg(feature = "posenet_decoder")]
        {
            Self::Posenet(Default::default())
        }
        #[cfg(not(feature = "posenet_decoder"))]
        {
            Self::HandRolled(Default::default())
        }
    }
}

impl Decoder for Decode {
    fn expected_output_tensors(&self) -> usize {
        match self {
            #[cfg(feature = "posenet_decoder")]
            Self::Posenet(d) => d.expected_output_tensors(),
            Self::HandRolled(d) => d.expected_output_tensors(),
        }
    }

    fn get_decoded_arrays(
        &self,
        interp: &mut tflite::Interpreter,
    ) -> Result<Box<[pose::Pose]>, Error> {
        match self {
            #[cfg(feature = "posenet_decoder")]
            Self::Posenet(d) => d.get_decoded_arrays(interp),
            Self::HandRolled(d) => d.get_decoded_arrays(interp),
        }
    }
}
