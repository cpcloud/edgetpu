use anyhow::Result;
use std::path::PathBuf;

const LIBS: [&str; 1] = ["opencv4"];

fn main() -> Result<()> {
    let include_paths = std::iter::once(PathBuf::from("src")).chain(
        LIBS.iter()
            .flat_map(|lib| pkg_config::Config::new().probe(lib).unwrap().include_paths),
    );

    autocxx_build::build("src/main.rs", include_paths, &[])
        .unwrap()
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .compile("tflite-pose");

    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=edgetpu_basic_engine");
    println!("cargo:rustc-link-lib=edgetpu_basic_engine_native");
    println!("cargo:rustc-link-lib=edgetpu_resource_manager");
    println!("cargo:rustc-link-lib=tensorflow-lite");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/wrapper.hpp");
    Ok(())
}
