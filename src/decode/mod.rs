use crate::{error::Error, pose, tflite};

pub(crate) trait Decoder {
    /// Return the number of expected_output_tensors the decoder expects to operate on.
    fn expected_output_tensors(&self) -> usize;

    /// Decode poses into a Vec of Pose.
    fn decode(
        &self,
        interp: &mut tflite::Interpreter,
        dims: (u16, u16),
    ) -> Result<Vec<pose::Pose>, Error>;

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
            Self::Posenet(posenet::Decoder::default())
        }
        #[cfg(not(feature = "posenet_decoder"))]
        {
            Self::HandRolled(hand_rolled::Decoder::default())
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

    fn decode(
        &self,
        interp: &mut tflite::Interpreter,
        dims: (u16, u16),
    ) -> Result<Vec<pose::Pose>, Error> {
        match self {
            #[cfg(feature = "posenet_decoder")]
            Self::Posenet(d) => d.decode(interp, dims),
            Self::HandRolled(d) => d.decode(interp, dims),
        }
    }
}
