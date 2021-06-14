#![feature(variant_count, never_type)]
use anyhow::Result;
use num_traits::cast::ToPrimitive;
use opencv::{
    core::{Mat, Point2f, Scalar, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::*,
    videoio::{VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2},
};

use std::path::PathBuf;
use structopt::StructOpt;

mod edgetpu;
mod engine;
mod error;
mod pose;
mod tflite;
mod tflite_sys;

#[cfg(not(target_arch = "aarch64"))]
fn draw_poses(
    poses: Vec<pose::Pose>,
    threshold: f32,
    timing: engine::Timing,
    out_frame: &mut Mat,
) -> Result<()> {
    for pose in poses.into_iter() {
        let xys = [None::<Point2f>; pose::NUM_KEYPOINTS];

        for keypoint in pose.keypoints {
            if keypoint.score >= threshold {
                let index = keypoint.kind.unwrap().idx()?;
                xys[index] = Some(keypoint.point);
                opencv::highgui::circle(
                    &mut out_frame,
                    keypoint.point,
                    6,
                    Scalar::from((0.0, 255.0, 0.0)),
                    -1,
                )?;
            }
        }

        for (a, b) in pose::KEYPOINT_EDGES {
            if let (Some(a_point), Some(b_point)) = (xys[a.idx()?], xys[b.idx()?]) {
                opencv::highgui::line(
                    &mut out_frame,
                    a_point,
                    b_point,
                    Scalar::from((0.0, 255.0, 255.0)),
                    2,
                );
            }
        }
    }
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn draw_poses(
    _poses: Vec<pose::Pose>,
    _threshold: f32,
    timing: engine::Timing,
    _out_frame: &mut Mat,
) {
}

#[derive(structopt::StructOpt)]
struct Opt {
    /// Path to a Tensorflow Lite model
    #[structopt()]
    model: PathBuf,

    /// /dev/videoDEVICE
    #[structopt(short, long, default_value = "0")]
    device: i32,

    /// Input image width
    #[structopt(short, long, default_value = "1281")]
    width: u16,

    /// Input image height
    #[structopt(short = "-H", long, default_value = "721")]
    height: u16,

    /// Width of the model image
    #[structopt(short, long, default_value = "1281")]
    model_width: u16,

    /// Height of the model image
    #[structopt(short = "-H", long, default_value = "721")]
    model_height: u16,

    /// Pose keypoint score threshold
    #[structopt(short, long, default_value = "0.2")]
    threshold: f32,
}

fn main() -> Result<!> {
    let opt = Opt::from_args();
    let threshold = opt.threshold;

    let mut capture = VideoCapture::new(opt.device, CAP_V4L2)?;
    capture.set(CAP_PROP_FRAME_WIDTH, opt.width.into())?;
    capture.set(CAP_PROP_FRAME_HEIGHT, opt.height.into())?;

    let mut in_frame = Mat::zeros(opt.width.into(), opt.height.into(), CV_8UC3)?.to_mat()?;
    let mut out_frame =
        Mat::zeros(opt.model_width.into(), opt.model_height.into(), CV_8UC3)?.to_mat()?;
    let out_frame_size = out_frame.size()?;

    let mut engine = engine::Engine::new(opt.model)?;

    loop {
        capture.read(&mut in_frame)?;

        resize(
            &in_frame,
            &mut out_frame,
            out_frame_size,
            0.0,
            0.0,
            INTER_LINEAR,
        )?;
        let (poses, timings) = engine.detect_poses(&out_frame)?;

        draw_poses(poses, threshold, timings, &mut out_frame)?;
    }
}
