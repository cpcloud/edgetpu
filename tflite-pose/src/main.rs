#![feature(variant_count, never_type)]
use anyhow::Result;
use opencv::{
    core::{Mat, Point2f, Point2i, Scalar, CV_8UC3},
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
    nframes: usize,
    width: u16,
) -> Result<()> {
    for pose in poses.into_iter() {
        let mut xys = [None::<Point2f>; pose::NUM_KEYPOINTS];

        // pub fn circle(img: &mut dyn core::ToInputOutputArray, center: core::Point, radius: i32, color: core::Scalar, thickness: i32, line_type: i32, shift: i32) -> Result<()>
        for keypoint in pose.keypoints {
            if keypoint.score >= threshold {
                let index = keypoint.kind.unwrap().idx()?;
                xys[index] = Some(keypoint.point);
                opencv::imgproc::circle(
                    out_frame,
                    Point2i::new(keypoint.point.x as i32, keypoint.point.y as i32),
                    6,
                    Scalar::from((0.0, 255.0, 0.0)),
                    -1,
                    -1,
                    -1,
                )?;
            }
        }

        // pub fn line(img: &mut dyn core::ToInputOutputArray, pt1: core::Point, pt2: core::Point, color: core::Scalar, thickness: i32, line_type: i32, shift: i32) -> Result<()>;
        for (a, b) in pose::KEYPOINT_EDGES {
            if let (Some(a_point), Some(b_point)) = (xys[a.idx()?], xys[b.idx()?]) {
                opencv::imgproc::line(
                    out_frame,
                    Point2i::new(a_point.x as i32, a_point.y as i32),
                    Point2i::new(b_point.x as i32, b_point.y as i32),
                    Scalar::from((0.0, 255.0, 255.0)),
                    2,
                    -1,
                    -1,
                )?;
            }
        }
    }

    let model_fps_text = format!(
        "Model FPS: {:.1}",
        nframes as f64 / timing.inference.as_secs_f64()
    );
    opencv::imgproc::put_text(
        out_frame,
        &model_fps_text,
        Point2i::new(i32::from(width) / 2, 15),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::from((38.0, 0.0, 255.0)),
        1,
        opencv::imgproc::LINE_AA,
        false,
    )?;
    let copy_fps_text = format!(
        "Copy FPS: {:.1}",
        nframes as f64 / timing.inference.as_secs_f64()
    );
    opencv::imgproc::put_text(
        out_frame,
        &copy_fps_text,
        Point2i::new(i32::from(width) / 2, 15),
        opencv::imgproc::FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::from((38.0, 0.0, 255.0)),
        1,
        opencv::imgproc::LINE_AA,
        false,
    )?;
    opencv::highgui::imshow("poses", out_frame)?;
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

    let mut nframes = 0;

    loop {
        capture.read(&mut in_frame)?;
        nframes += 1;

        resize(
            &in_frame,
            &mut out_frame,
            out_frame_size,
            0.0,
            0.0,
            INTER_LINEAR,
        )?;

        let (poses, timings) = engine.detect_poses(&out_frame)?;

        draw_poses(
            poses,
            threshold,
            timings,
            &mut out_frame,
            nframes,
            opt.width,
        )?;
    }
}
