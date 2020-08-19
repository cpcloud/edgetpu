mod tflite_sys;

use anyhow::Result;
use structopt::StructOpt;
use tflite_sys::*;

#[derive(structopt::StructOpt)]
struct Args {
    #[structopt(required = true)]
    model_path: std::path::PathBuf,
}

fn main() -> Result<()> {
    let args = Args::from_args();
    let cstring =
        std::ffi::CString::new(args.model_path.to_string_lossy().to_string().into_bytes())?;
    unsafe {
        let reporter = tflite_DefaultErrorReporter();
        tflite_FlatBufferModel::BuildFromFile(cstring.as_ptr(), reporter)
    };
    Ok(())
}
