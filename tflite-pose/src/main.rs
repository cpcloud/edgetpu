use anyhow::{anyhow, Result};
use structopt::StructOpt;

use std::os::unix::ffi::OsStrExt;

#[cxx::bridge(namespace = pose)]
mod ffi {
    extern "C" {
        include!("wrapper.h");

        type FlatBufferModel;
        type Interpreter;

        fn build_model_from_file(model_path: &str) -> UniquePtr<FlatBufferModel>;
        fn build_interpreter(model: &mut FlatBufferModel) -> UniquePtr<Interpreter>;
        fn allocate_tensors(interpreter: &mut Interpreter);
    }
}

#[derive(structopt::StructOpt)]
struct Args {
    #[structopt(required = true)]
    model_path: std::path::PathBuf,
}

fn main() -> Result<()> {
    let args = Args::from_args();
    let path = args.model_path.to_string_lossy().to_string();
    let mut model = ffi::build_model_from_file(&path);
    if model.is_null() {
        return Err(anyhow!(
            "unable to load model {}",
            args.model_path.display()
        ));
    }
    let mut interp = ffi::build_interpreter(&mut model);
    ffi::allocate_tensors(&mut interp);
    Ok(())
}
