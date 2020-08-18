mod tflite_sys;

use anyhow::Result;
use structopt::StructOpt;
use tflite_sys::root::tflite;

#[derive(structopt::StructOpt)]
struct Args {
    #[structopt(required = true)]
    model_path: std::path::PathBuf,
}

fn main() -> Result<()> {
    let args = Args::from_args();
    let cstring = std::ffi::CString::new(args.model_path.display().to_string().as_bytes())?;
    let ptr = cstring.as_ptr();
    unsafe {
        let reporter = tflite::DefaultErrorReporter();
        tflite::FlatBufferModel::BuildFromFile(ptr, reporter)
    };
    Ok(())
}
