use anyhow::{anyhow, Result};
use structopt::StructOpt;

#[cxx::bridge(namespace = pose)]
mod ffi {
    extern "C" {
        include!("wrapper.h");

        type BasicEngine;

        fn build_engine(model_path: &str) -> UniquePtr<BasicEngine>;
        fn run_inference(engine: UniquePtr<BasicEngine>, data: &[u8]) -> Vec<f32>;
    }
}

#[derive(structopt::StructOpt)]
struct Args {
    #[structopt(required = true)]
    model_path: std::path::PathBuf,
}

fn main() -> Result<()> {
    let args = Args::from_args();
    let model_path = args.model_path.to_string_lossy().to_string();
    let engine = ffi::build_engine(&model_path);
    Ok(())
}
