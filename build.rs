use std::env;
use std::path::PathBuf;
use anyhow::Result;

fn main() -> Result<()> {
    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=tensorflow-lite");
    println!("cargo:rerun-if-changed=wrapper.h");

    bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()?;
        .write_to_file(PathBuf::from(env::var("OUT_DIR")?).join("bindings.rs"))?;
    Ok(())
}
