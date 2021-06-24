#![feature(variant_count)]

use anyhow::{Context, Result};
use error::Error;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::ArrayView3;
use num_traits::cast::ToPrimitive;
use opencv::{
    core::{Mat, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::{MatExprTrait, MatTrait, MatTraitManual, VideoCaptureTrait},
    videoio::{VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2},
};
use std::{
    convert::TryFrom,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use structopt::StructOpt;

mod coral;
mod decode;
mod edgetpu;
mod engine;
mod error;
mod pose;
mod tflite;
mod tflite_sys;

fn draw_poses(
    #[cfg(feature = "gui")] poses: &[pose::Pose],
    #[cfg(not(feature = "gui"))] _poses: &[pose::Pose],
    #[cfg(feature = "gui")] threshold: f32,
    #[cfg(not(feature = "gui"))] _threshold: f32,
    timing: engine::Timing,
    #[cfg(feature = "gui")] out_frame: &mut Mat,
    #[cfg(not(feature = "gui"))] _out_frame: &mut Mat,
    frame_duration: Duration,
    nframes: usize,
    pb_model_cam_fps: &indicatif::ProgressBar,
) -> Result<(), Error> {
    let nframes = nframes.to_f64().ok_or(Error::ConvertToF64)?;
    let fps_text = format!(
        "FPS => model: {:.1}, cam: {:.1}",
        nframes / timing.inference.as_secs_f64(),
        nframes / frame_duration.as_secs_f64()
    );

    #[cfg(feature = "gui")]
    {
        use opencv::{
            core::Scalar,
            imgproc::{FONT_HERSHEY_SIMPLEX, LINE_8, LINE_AA},
        };

        const GREEN: (f64, f64, f64) = (0.0, 255.0, 0.0);
        const YELLOW: (f64, f64, f64) = (0.0, 255.0, 255.0);
        const WHITE: (f64, f64, f64) = (255.0, 255.0, 255.0);

        for pose in poses {
            let mut xys = [None; pose::NUM_KEYPOINTS];

            pose.keypoints
                .iter()
                .filter_map(|&pose::Keypoint { kind, score, point }| {
                    kind.and_then(|kind| {
                        if score >= threshold {
                            Some((kind, point))
                        } else {
                            None
                        }
                    })
                })
                .try_for_each(|(kind, point)| {
                    let index = kind.idx()?;
                    xys[index] = Some(point);
                    opencv::imgproc::circle(
                        out_frame,
                        point.to().ok_or(Error::ConvertPoint2fToPoint2i(point))?,
                        6,
                        Scalar::from(GREEN),
                        1,      // thickness
                        LINE_8, // line_type
                        0,      // shift
                    )
                    .map_err(Error::DrawCircle)
                })?;

            for (a, b) in pose::constants::KEYPOINT_EDGES {
                if let (Some(a_point), Some(b_point)) = (xys[a.idx()?], xys[b.idx()?]) {
                    opencv::imgproc::line(
                        out_frame,
                        a_point
                            .to()
                            .ok_or(Error::ConvertPoint2fToPoint2i(a_point))?,
                        b_point
                            .to()
                            .ok_or(Error::ConvertPoint2fToPoint2i(b_point))?,
                        Scalar::from(YELLOW),
                        2,      // thickness
                        LINE_8, // line_type
                        0,      // shift
                    )
                    .map_err(Error::DrawLine)?;
                }
            }
        }
        opencv::imgproc::put_text(
            out_frame,
            &fps_text,
            opencv::core::Point2i::new(0, 15),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar::from(WHITE),
            1,       // thickness
            LINE_AA, // line_type
            false,   // bottom_left_origin
        )
        .map_err(Error::PutText)?;
        opencv::highgui::imshow("poses", out_frame).map_err(Error::ImShow)?;
    }

    pb_model_cam_fps.set_message(fps_text);
    pb_model_cam_fps.inc(1);
    Ok(())
}

#[cfg(feature = "gui")]
fn wait_q(delay_ms: i32) -> Result<bool> {
    const Q_KEY: u8 = b'q';
    Ok(opencv::highgui::wait_key(delay_ms)? != i32::from(Q_KEY))
}

#[cfg(not(feature = "gui"))]
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
    #[structopt(short = "-W", long, default_value = "1")]
    wait_key_ms: i32,

    #[structopt(subcommand)]
    decoder: decode::Decode,
}

fn mat_to_ndarray(input: &Mat) -> Result<ArrayView3<u8>, Error> {
    let step = input.step1(0).map_err(Error::GetStep1)?
        * input.elem_size1().map_err(Error::GetElemSize1)?;
    let rows = usize::try_from(input.rows()).map_err(Error::ConvertRowsToUsize)?;
    let num_elements = step * rows;

    // copy the bytes into the input tensor
    let raw_data = input.data().map_err(Error::GetMatData)? as _;
    let data = unsafe { std::slice::from_raw_parts(raw_data, num_elements) };
    ArrayView3::from_shape(
        (
            usize::try_from(input.rows()).map_err(Error::ConvertDimI32ToUSize)?,
            usize::try_from(input.cols()).map_err(Error::ConvertDimI32ToUSize)?,
            usize::try_from(input.channels().map_err(Error::GetChannels)?)
                .map_err(Error::ConvertDimI32ToUSize)?,
        ),
        data,
    )
    .map_err(Error::ConstructNDArrayFromMat)
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
    println!(
        "width: {}, height: {}",
        width.to_usize().unwrap(),
        height.to_usize().unwrap()
    );

    let mut in_frame = Mat::zeros(
        height.to_i32().expect("failed to convert height to i32"),
        width.to_i32().expect("failed to convert width to i32"),
        CV_8UC3,
    )?
    .to_mat()?;
    let mut out_frame = Mat::zeros(opt.height.into(), opt.width.into(), CV_8UC3)?.to_mat()?;
    let out_frame_size = out_frame.size()?;

    let mut engine = engine::Engine::new(opt.model, opt.decoder)?;

    let mut nframes = 0;
    let mut frame_duration = Default::default();

    let pb_model_cam_fps = ProgressBar::new_spinner().with_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template("{prefix:.bold.dim} {spinner} {wide_msg}"),
    );

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    while wait_q(opt.wait_key_ms).context("failed waiting for 'q' key")?
        && running.load(Ordering::SeqCst)
    {
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

        let (poses, timing) = engine.detect_poses(mat_to_ndarray(&out_frame)?)?;
        draw_poses(
            &poses,
            threshold,
            timing,
            &mut out_frame,
            frame_duration,
            nframes,
            &pb_model_cam_fps,
        )?;
    }
    Ok(())
}
