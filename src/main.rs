#![feature(variant_count)]

use anyhow::{anyhow, Context, Result};
use error::Error;
use indicatif::{ProgressBar, ProgressStyle};
use num_traits::cast::ToPrimitive;
use opencv::{
    core::{Mat, Mat_AUTO_STEP, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::{MatExprTrait, MatTrait, MatTraitManual, VideoCaptureTrait},
    videoio::{VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2},
};
use std::{
    convert::TryFrom,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::channel,
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use structopt::StructOpt;
use tracing::{info, trace};
use tracing_subscriber::layer::SubscriberExt;

mod coral;
mod decode;
mod edgetpu;
mod engine;
mod error;
mod ffi;
mod pose;
mod tflite;
mod tflite_sys;

#[allow(clippy::too_many_arguments)]
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
    pb_model_cam_fps: Option<&indicatif::ProgressBar>,
) -> Result<(), Error> {
    let nframes = nframes.to_f64().ok_or(Error::ConvertToF64)?;
    let fps_text = format!(
        "FPS => model: {:.1}, cam: {:.1}",
        nframes / timing.inference.as_secs_f64(),
        nframes / frame_duration.as_secs_f64(),
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

    if let Some(pb_model_cam_fps) = pb_model_cam_fps {
        pb_model_cam_fps.set_message(fps_text);
        pb_model_cam_fps.inc(1);
    }

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

/// Convert an input Mat to a slice of bytes.
fn mat_to_slice(input: &Mat) -> Result<&[u8], Error> {
    let step = input.step1(0).map_err(Error::GetStep1)?
        * input.elem_size1().map_err(Error::GetElemSize1)?;
    let rows = usize::try_from(input.rows()).map_err(Error::ConvertRowsToUsize)?;
    let num_elements = step * rows;

    // copy the bytes into the input tensor
    let raw_data = input.data().map_err(Error::GetMatData)? as _;
    Ok(unsafe { std::slice::from_raw_parts(raw_data, num_elements) })
}

#[derive(structopt::StructOpt)]
struct Opt {
    /// Path to a Tensorflow Lite edgetpu model.
    #[structopt(required = true)]
    models: Vec<PathBuf>,

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

    #[structopt(short = "-W", long, default_value = "1")]
    wait_key_ms: i32,

    #[structopt(short, long, default_value = "info", env = "RUST_LOG")]
    log_level: tracing_subscriber::filter::EnvFilter,

    #[structopt(short, long, default_value = "1000")]
    input_queue_size: usize,

    #[structopt(short, long, default_value = "1000")]
    output_queue_size: usize,

    #[structopt(subcommand)]
    decoder: decode::Decode,

    #[structopt(short, long)]
    show_progress: bool,
}

fn main() -> Result<()> {
    let opt = Opt::from_args();
    let threshold = opt.threshold;

    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(opt.log_level),
    )?;

    let mut capture = VideoCapture::new(opt.device, CAP_V4L2)?;

    if let Some(width) = opt.frame_width.map(f64::from) {
        capture.set(CAP_PROP_FRAME_WIDTH, width)?;
    }

    if let Some(height) = opt.frame_height.map(f64::from) {
        capture.set(CAP_PROP_FRAME_HEIGHT, height)?;
    }

    let width = capture.get(CAP_PROP_FRAME_WIDTH)?;
    let height = capture.get(CAP_PROP_FRAME_HEIGHT)?;

    info!(
        message = "got dimensions from video capture",
        width = width.to_i64().unwrap(),
        height = height.to_i64().unwrap()
    );

    let mut in_frame = Mat::zeros(
        height.to_i32().expect("failed to convert height to i32"),
        width.to_i32().expect("failed to convert width to i32"),
        CV_8UC3,
    )?
    .to_mat()
    .context("failed converting input frame MatExpr to Mat")?;
    let mut out_frame = Mat::zeros(opt.height.into(), opt.width.into(), CV_8UC3)
        .context("failed to construct MatExpr of zeros")?
        .to_mat()
        .context("failed convertin output frame MatExpr to Mat")?;
    let out_frame_size = out_frame
        .size()
        .context("failed getting output frame dimensions")?;

    let running = Arc::new(AtomicBool::new(true));
    let running_ctrl_c = running.clone();

    ctrlc::set_handler(move || {
        running_ctrl_c.store(false, Ordering::SeqCst);
    })
    .context("failed setting Ctrl-C handler")?;

    let engine = Arc::new(
        engine::Engine::new(
            &opt.models,
            opt.decoder,
            opt.input_queue_size,
            opt.output_queue_size,
        )
        .context("failed constructing engine")?,
    );

    let frame_duration = Arc::new(Mutex::new(Duration::default()));
    let frame_duration_push = frame_duration.clone();

    let pb_model_cam_fps = if opt.show_progress {
        Some(
            ProgressBar::new_spinner().with_style(
                ProgressStyle::default_spinner()
                    .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
                    .template("{prefix:.bold.dim} {spinner} {wide_msg}"),
            ),
        )
    } else {
        None
    };

    let (poses_tx, poses_rx) = channel();
    let (input_frames_tx, input_frames_rx) = channel();

    let wait_key_ms = opt.wait_key_ms;
    let running_push = running.clone();
    let engine_push = engine.clone();

    crossbeam::thread::scope(|scope| {
        scope.spawn(move |_| {
            while running_push.load(Ordering::SeqCst) {
                let frame_start = Instant::now();
                if !capture
                    .read(&mut in_frame)
                    .context("failed reading frame")?
                {
                    return Err(anyhow!("reading frame returned false"));
                }
                *frame_duration_push.lock().unwrap() += frame_start.elapsed();

                resize(
                    &in_frame,
                    &mut out_frame,
                    out_frame_size,
                    0.0,
                    0.0,
                    INTER_LINEAR,
                )
                .context("failed to resize frame")?;

                let image_slice = mat_to_slice(&out_frame)?;
                let inputs = Arc::new(vec![engine_push.alloc_input_tensor(image_slice)?]);

                let image_vec = image_slice.to_vec();
                engine_push.push(Some(inputs.clone()))?;
                input_frames_tx.send((inputs, image_vec))?;
            }

            Ok(())
        });

        let running_pop = running.clone();
        scope.spawn(move |_| {
            while running_pop.load(Ordering::SeqCst) {
                let (_input_tensors, image_bytes) = input_frames_rx.recv()?;
                let _output_tensors = engine.pop().context("failed to pop")?;
                let poses = engine.decode_poses().context("failed detecting poses")?;
                poses_tx.send((poses, engine.timing(), engine.frame_num(), image_bytes))?;
            }
            Ok::<_, anyhow::Error>(())
        });

        scope.spawn(move |_| {
            while let Ok((poses, timing, frame_num, mut image_bytes)) = poses_rx.recv() {
                if !wait_q(wait_key_ms).context("failed waiting for 'q' key")? {
                    running.store(false, Ordering::SeqCst);
                }

                if !running.load(Ordering::SeqCst) {
                    break;
                }

                let mut mat = unsafe {
                    Mat::new_size_with_data(
                        out_frame_size,
                        CV_8UC3,
                        image_bytes.as_mut_slice().as_mut_ptr().cast(),
                        Mat_AUTO_STEP,
                    )?
                };

                trace!(
                    zero_bytes = ?image_bytes
                        .iter()
                        .fold(0.0, |c, &byte| c + f64::from(u8::from(byte == 0)))
                        / image_bytes.len() as f64
                );

                draw_poses(
                    &poses,
                    threshold,
                    timing,
                    &mut mat,
                    *frame_duration.lock().unwrap(),
                    frame_num,
                    pb_model_cam_fps.as_ref(),
                )
                .context("failed drawing poses")?;
            }
            Ok::<_, anyhow::Error>(())
        });

        Ok(())
    })
    .unwrap()
}
