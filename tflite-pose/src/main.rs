#![feature(variant_count, never_type)]

use autocxx::include_cpp;

include_cpp! {
    #include "wrapper.hpp"
    safety!(unsafe)
    generate!("pose::Engine")
}

use anyhow::Result;
use opencv::{
    core::{Mat, Point2f, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::*,
    videoio::{VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2},
};
use std::{convert::TryFrom, path::PathBuf, time::Duration};
use structopt::StructOpt;

const NANOS_PER_MILLI: f64 = 1_000_000.0;

#[derive(Debug, thiserror::Error)]
enum Error {
    #[error("failed to convert inference milliseconds to duration")]
    ConvertInferenceMillis(#[source] std::num::TryFromIntError),

    #[error("failed to get typed data from OpenCV Mat")]
    GetTypedData(#[source] opencv::Error),
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
            kind: KeypointKind::Nose,
            point: Point2f::new(0.0_f32, 0.0_f32),
            score: 0.0,
        }
    }
}

struct Pose {
    keypoints: [Keypoint; std::mem::variant_count::<KeypointKind>()],
    score: f64,
}

struct Engine {
    engine: cxx::UniquePtr<ffi::pose::Engine>,
    input_tensor_shape: Vec<usize>,
    output_offsets: Vec<usize>,
    mirror: bool,
}

impl Engine {
    fn new(model_path: &str, mirror: bool) -> Self {
        let engine = ffi::pose::Engine::make_unique(model_path);
        let input_tensor_shape = engine
            .get_input_tensor_shape()
            .into_iter()
            .map(|&value| value as usize)
            .collect::<Vec<_>>();
        let output_offsets = engine
            .get_all_output_tensors_sizes()
            .iter()
            .copied()
            .scan(0, |result, shape| {
                *result += shape;
                Some(*result)
            })
            .collect();
        Self {
            engine,
            input_tensor_shape,
            output_offsets,
            mirror,
        }
    }

    fn width(&self) -> usize {
        self.input_tensor_shape[2]
    }

    fn depth(&self) -> usize {
        self.input_tensor_shape[3]
    }

    fn detect_poses(&mut self, input: &Mat) -> Result<(Vec<Pose>, Duration), Error> {
        let bytes = input
            .data_typed::<u8>()
            .map_err(Error::GetTypedData)?
            .to_owned();

        let sizes = self.engine.get_all_output_tensors_sizes();
        let mut keypoints = vec![0; sizes[0]];
        let mut keypoint_scores = vec![0; sizes[1]];
        let mut pose_scores = vec![0; sizes[2]];
        let nposes = self.engine.run_inference(bytes);
        let inference_time = Duration::from_nanos(
            (f64::from(self.engine.get_inference_time()) * NANOS_PER_MILLI) as u64,
        );

        Ok((vec![], inference_time))
    }
}

const EDGES: [(KeypointKind, KeypointKind); 19] = [
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
    model: String,

    /// /dev/videoDEVICE
    #[structopt(short, long, default_value = "0")]
    device: i32,

    /// Width
    #[structopt(short, long, default_value = "1280")]
    width: usize,

    /// Height
    #[structopt(short = "-H", long, default_value = "720")]
    height: u16,

    /// Mirror
    #[structopt(short, long)]
    mirror: bool,

    /// Pose keypoint score threshold
    #[structopt(short, long, default_value = "0.2")]
    threshold: f64,
}

fn main() -> Result<!> {
    let opt = Opt::from_args();

    let mut capture = VideoCapture::new(opt.device, CAP_V4L2)?;
    capture.set(CAP_PROP_FRAME_WIDTH, 1920.0);
    capture.set(CAP_PROP_FRAME_HEIGHT, 1080.0);

    let mut in_frame = Mat::zeros(1920, 1080, CV_8UC3)?.to_mat()?;
    let mut out_frame = Mat::zeros(1280, 720, CV_8UC3)?.to_mat()?;

    let engine = Engine::new(&opt.model, opt.mirror);

    loop {
        capture.read(&mut in_frame)?;
        resize(
            &in_frame,
            &mut out_frame,
            out_frame.size()?,
            0.0,
            0.0,
            INTER_LINEAR,
        )?;
        let (poses, _) = engine.detect_poses(&out_frame)?;
    }
}
