use anyhow::{anyhow, Result};
use structopt::StructOpt;

#[cxx::bridge(namespace = pose)]
mod ffi {
    extern "C" {
        include!("wrapper.h");

        type FlatBufferModel;
        type Interpreter;

        fn build_model_from_file(model_path: &str) -> UniquePtr<FlatBufferModel>;
        fn build_interpreter(model: &FlatBufferModel) -> UniquePtr<Interpreter>;
        fn num_inputs(interp: &Interpreter) -> usize;
        fn input_name(interp: &Interpreter, index: usize) -> String;
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
    let model = ffi::build_model_from_file(&path);
    let interpreter = ffi::build_interpreter(&model);

    if interpreter.is_null() {
        return Err(anyhow!("Foo!"));
    }

    // for i in 0..ffi::num_inputs(&interpreter) {
    //     println!("input: {}, name: {}", i, ffi::input_name(&interpreter, i));
    // }
    Ok(())
}
