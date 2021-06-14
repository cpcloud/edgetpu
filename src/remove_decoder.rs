use anyhow::{Context, Result};
use num_traits::cast::ToPrimitive;
use opencv::{
    core::{Mat, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::*,
    videoio::{VideoCapture, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_V4L2},
};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(structopt::StructOpt)]
struct Opt {
    /// Path to a Tensorflow Lite edgetpu model.
    #[structopt(short, long, required = true)]
    input: PathBuf,

    /// Path to a Tensorflow Lite edgetpu model.
    #[structopt(short, long, required = true)]
    output: PathBuf,

    #[structopt(short, long, required = true)]
    schema: PathBuf,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let opt = Opt::from_args();
    let model_path = opt.input;
    let model_basename = model_path;
    Ok(())
}
