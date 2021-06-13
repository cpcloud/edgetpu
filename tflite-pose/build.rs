use anyhow::Result;
use std::path::PathBuf;

const LIBS: [&str; 3] = ["opencv4", "edgetpu", "tensorflow-lite"];

fn main() -> Result<()> {
    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=tensorflow-lite");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/wrapper.h");

    let mut bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_arg("-O3");
    for include_path in LIBS
        .iter()
        .flat_map(|lib| pkg_config::Config::new().probe(lib).unwrap().include_paths)
    {
        bindings = bindings.clang_arg(format!("-I{}", include_path.display()));
    }

    bindings
        .generate()
        .unwrap()
        .write_to_file(PathBuf::from(std::env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}
