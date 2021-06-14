#![feature(variant_count, never_type)]
use anyhow::{anyhow, Result};
use error::Error;
use opencv::{
    core::{Mat, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::*,
    videoio::{VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2},
};
use std::{
    convert::TryFrom,
    path::{Path, PathBuf},
    time::Duration,
};
use structopt::StructOpt;

mod edgetpu;
mod engine;
mod error;
mod pose;
mod tflite;
mod tflite_sys;

#[derive(structopt::StructOpt)]
struct Opt {
    /// Path to a Tensorflow Lite model
    #[structopt()]
    model: PathBuf,

    /// /dev/videoDEVICE
    #[structopt(short, long, default_value = "0")]
    device: i32,

    /// Width
    #[structopt(short, long, default_value = "1281")]
    width: usize,

    /// Height
    #[structopt(short = "-H", long, default_value = "721")]
    height: u16,

    /// Pose keypoint score threshold
    #[structopt(short, long, default_value = "0.2")]
    threshold: f64,
}

fn main() -> Result<!> {
    let opt = Opt::from_args();

    let devices = edgetpu::Devices::new()?;
    println!("num_devices: {}", devices.len());

    for device in devices.iter() {
        println!("device.path(): {}", device?.path()?);
    }

    let mut capture = VideoCapture::new(opt.device, CAP_V4L2)?;
    capture.set(CAP_PROP_FRAME_WIDTH, 1920.0)?;
    capture.set(CAP_PROP_FRAME_HEIGHT, 1080.0)?;

    let mut in_frame = Mat::zeros(1920, 1080, CV_8UC3)?.to_mat()?;
    let mut out_frame = Mat::zeros(1280, 720, CV_8UC3)?.to_mat()?;

    let engine = engine::Engine::new(opt.model)?;

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
