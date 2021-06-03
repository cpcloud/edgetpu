#![feature(variant_count, never_type)]

use anyhow::{anyhow,Context,Result};
use opencv::imgproc::prelude::*;
use opencv::core::prelude::*;
use opencv::videoio::prelude::*;
use std::time::{Instant,Duration};
use std::path::PathBuf;
use structopt::StructOpt;

autocxx::include_cpp! {
    #include "src/cpp/basic/basic_engine.h"
    generate!("coral::BasicEngine")
    safety!(unsafe_ffi)
}

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("foo")]
    InputDimensions,

    #[error("bar")]
    FirstDimensionSize,

    #[error("baz")]
    DepthSize,
}

#[derive(Debug, Copy, Clone)]
enum KeypointKind {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
}

#[derive(Debug, Copy, Clone)]
struct Keypoint {
    kind: KeypointKind,
    point: Point2f,
    score: f64,
}

impl Default for Keypoint {
    fn default() -> Self {
        Self {
            kind: Self::Nose,
            point: Point2f::new(0.0_f32, 0.0_f32),
            score: 0.0,
        }
    }
}

const NUM_KEYPOINTS: usize = std::mem::variant_count::<KeypointKind>();

type Keypoints = [Keypoint; NUM_KEYPOINTS];

struct Pose {
    keypoints: Keypoints,
    score: f64,
}

type Poses = Vec<Pose>;

struct Engine {
    engine: ffi::coral::BasicEngine,
    input_tensor_shape: Vec<i32>,
    output_offsets: Vec<usize>,
    mirror: bool,
}

impl Engine {
    fn detect_poses(&mut self, input: &Mat) -> Result<(Poses, Duration), Error> {
        let outputs = self.engine.RunInference(input);
        let inference_time = engine.get_inference_time();
        let mut poses = vec![];

        for (pose_i, keypoint) in keypoints.iter().enumerate() {
            let mut keypoint_map = Default::default();

            for (point_i, (column, row)) in keypoint.iter().enumerate() {
                let mut kp = Keypoint {
                    kind,
                    (row, column),
                    keypoint_scores[(pose_i, point_i)],
                };

                if self.mirror {
                    kp.point.x = self.width() - kp.point.x
                }

                keypoint_map[point_i] = kp;
            }
            poses.push(Pose { keypoints: keypoint_map, score: pose_scores[pose_i] });
        }

        Ok((poses, Duration::from_millis(u64::try_from(inference_time).map_err(Error::ConvertInferenceMillis)?)))
    }
}

const EDGES: &[(Keypoint, Keypoint); 19] = [
    (KeypointKind::Nose, KeypointKind::LeftEye),
    (KeypointKind::Nose, KeypointKind::RightEye),
    (KeypointKind::Nose, KeypointKind::LeftEar),
    (KeypointKind::Nose, KeypointKind::RightEar),
    (KeypointKind::LeftEar, KeypointKind::LeftEye),
    (KeypointKind::RightEar, KeypointKind::RightEye),
    (KeypointKind::LeftEye, KeypointKind::RightEye),
    (KeypointKind::LeftShoulder, KeypointKind::RightShoulder),
    (KeypointKind::LeftShoulder, KeypointKind::LeftElbow),
    (KeypointKind::LeftShoulder, KeypointKind::LeftHip),
    (KeypointKind::RightShoulder, KeypointKind::RightElbow),
    (KeypointKind::RightShoulder, KeypointKind::RightHip),
    (KeypointKind::LeftElbow, KeypointKind::LeftWrist),
    (KeypointKind::RightElbow, KeypointKind::RightWrist),
    (KeypointKind::LeftHip, KeypointKind::RightHip),
    (KeypointKind::LeftHip, KeypointKind::LeftKnee),
    (KeypointKind::RightHip, KeypointKind::RightKnee),
    (KeypointKind::LeftKnee, KeypointKind::LeftAnkle),
    (KeypointKind::RightKnee, KeypointKind::RightAnkle),
];

#[derive(structopt::StructOpt)]
struct Opt {
    /// Path to a Tensorflow Lite model
    #[structopt(required = true)]
    model: PathBuf,

    /// Path to a v4l2 compatible device
    #[structopt(required = true)]
    device: PathBuf,

    /// Width
    #[structopt(required = true)]
    width: i32,

    /// Height
    #[structopt(required = true)]
    height: i32,

    /// Pose keypoint score threshold
    #[structopt(default_value = "0.2")]
    threshold: f64,
}

fn main() -> Result<!> {
    let mut capture = cv::VideoCapture::new(0, cv::CAP_V4L2)?;
    capture.set(CAP_PROP_FRAME_WIDTH, 1920.0);
    capture.set(CAP_PROP_FRAME_HEIGHT, 1080.0);

    let mut in_frame = Mat::new_rows_cols(1920, 1080, CV_8UC3)?;
    let mut out_frame = Mat::new_rows_cols(1280, 720, CV_8UC3)?;

    loop {
        capture.read(&mut in_frame)?;
        resize(&in_frame, &mut out_frame, out_frame.size(), 0, 0, INTER_LINEAR)?;
        let (poses, _) = engine.detect_poses(out_frame)?;
    }
}
