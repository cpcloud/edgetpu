#![feature(variant_count)]

use anyhow::{Context, Result};
use num_traits::cast::ToPrimitive;
use opencv::{
    core::{Mat, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::*,
    videoio::{VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
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
    frame_duration: Duration,
    nframes: usize,
    width: u16,
) -> Result<()> {
    use opencv::{
        core::{Point2i, Scalar},
        imgproc::{FONT_HERSHEY_SIMPLEX, LINE_8, LINE_AA},
    };

    for pose in poses.into_iter() {
        let mut xys = [None; pose::NUM_KEYPOINTS];

        for keypoint in pose.keypoints {
            if keypoint.score >= threshold && keypoint.kind.is_some() {
                let index = keypoint.kind.unwrap().idx()?;
                xys[index] = Some(keypoint.point);
                opencv::imgproc::circle(
                    out_frame,
                    Point2i::new(keypoint.point.x as i32, keypoint.point.y as i32),
                    6,
                    Scalar::from((0.0, 255.0, 0.0)),
                    1,      // thickness
                    LINE_8, // line_type
                    0,      // shift
                )?;
            }
        }

        for (a, b) in pose::KEYPOINT_EDGES {
            if let (Some(a_point), Some(b_point)) = (xys[a.idx()?], xys[b.idx()?]) {
                opencv::imgproc::line(
                    out_frame,
                    Point2i::new(a_point.x as i32, a_point.y as i32),
                    Point2i::new(b_point.x as i32, b_point.y as i32),
                    Scalar::from((0.0, 255.0, 255.0)),
                    2,      // thickness
                    LINE_8, // line_type
                    0,      // shift
                )?;
            }
        }
    }

    opencv::imgproc::put_text(
        out_frame,
        &format!(
            "Model FPS: {:.1}",
            nframes as f64 / timing.inference.as_secs_f64()
        ),
        Point2i::new(i32::from(width) / 2, 15),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::from((38.0, 0.0, 255.0)),
        1,       // thickness
        LINE_AA, // line_type
        false,   // bottom_left_origin
    )?;
    opencv::imgproc::put_text(
        out_frame,
        &format!(
            "Cam FPS: {:.1}",
            nframes as f64 / frame_duration.as_secs_f64()
        ),
        Point2i::new(i32::from(width) / 2, 30),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar::from((38.0, 0.0, 255.0)),
        1,       // thickness
        LINE_AA, // line_type
        false,   // bottom_left_origin
    )?;
    opencv::highgui::imshow("poses", out_frame)?;
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn draw_poses(
    _poses: Vec<pose::Pose>,
    _threshold: f32,
    _timing: engine::Timing,
    _out_frame: &mut Mat,
    _frame_duration: Duration,
    _nframes: usize,
    _width: u16,
) -> Result<()> {
    Ok(())
}

#[cfg(not(target_arch = "aarch64"))]
fn wait_q(delay_ms: i32) -> Result<bool> {
    const Q_KEY: u8 = b'q';
    Ok(opencv::highgui::wait_key(delay_ms)? != i32::from(Q_KEY))
}

#[cfg(target_arch = "aarch64")]
fn wait_q(_delay_ms: i32) -> Result<bool> {
    Ok(true)
}

#[derive(structopt::StructOpt)]
struct Opt {
    /// Path to a Tensorflow Lite edgetpu model.
    #[structopt()]
    model: PathBuf,

    /// A v42l compatible device: /dev/videoDEVICE
    #[structopt(short, long, default_value = "0")]
    device: i32,

    /// The width of the image the model expects.
    #[structopt(short, long, default_value = "641")]
    width: u16,

    /// The height of the image the model expects.
    #[structopt(short = "-H", long, default_value = "481")]
    height: u16,

    /// Pose keypoint score threshold.
    #[structopt(short, long, default_value = "0.2")]
    threshold: f32,

    /// The width of the input frame.
    #[structopt(long)]
    frame_width: Option<u16>,

    /// The height of the input frame.
    #[structopt(long)]
    frame_height: Option<u16>,

    // TH
    #[structopt(short, long, default_value = "1")]
    wait_key_ms: i32,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let threshold = opt.threshold;

    let mut capture = VideoCapture::new(opt.device, CAP_V4L2)?;

    if let Some(width) = opt.frame_width.map(f64::from) {
        capture.set(CAP_PROP_FRAME_WIDTH, width)?;
    }

    if let Some(height) = opt.frame_height.map(f64::from) {
        capture.set(CAP_PROP_FRAME_HEIGHT, height)?;
    }

    let width = capture.get(CAP_PROP_FRAME_WIDTH)?;
    let height = capture.get(CAP_PROP_FRAME_HEIGHT)?;
    println!("width: {}, height: {}", width, height);

    let mut in_frame = Mat::zeros(
        height.to_i32().expect("failed to convert height to i32"),
        width.to_i32().expect("failed to convert width to i32"),
        CV_8UC3,
    )?
    .to_mat()?;
    let mut out_frame = Mat::zeros(opt.height.into(), opt.width.into(), CV_8UC3)?.to_mat()?;
    let out_frame_size = out_frame.size()?;

    let mut engine = engine::Engine::new(opt.model)?;

    let mut nframes = 0;
    let mut frame_duration = Default::default();

    while wait_q(opt.wait_key_ms).context("failed waiting for 'q' key")? {
        let frame_start = Instant::now();
        capture.read(&mut in_frame)?;
        frame_duration += frame_start.elapsed();
        nframes += 1;

        resize(
            &in_frame,
            &mut out_frame,
            out_frame_size,
            0.0,
            0.0,
            INTER_LINEAR,
        )?;

        let poses = engine.detect_poses(&out_frame)?;
        draw_poses(
            poses,
            threshold,
            engine.timing,
            &mut out_frame,
            frame_duration,
            nframes,
            opt.width,
        )?;
        #[cfg(not(target_arch = "aarch64"))]
        opencv::highgui::imshow("poses", &out_frame)?;
    }
    Ok(())
}
