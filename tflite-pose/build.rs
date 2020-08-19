use anyhow::Result;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rustc-link-lib=tensorflow-lite");
    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=usb-1.0");
    println!("cargo:rerun-if-changed=wrapper.h");

    bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_arg("-x")
        .clang_arg("c++")
        .whitelist_type("tflite::.*")
        .whitelist_var("tflite::.*")
        .whitelist_function("tflite::.*")
        .opaque_type("std::.*")
        .generate()
        .unwrap()
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))?;
    Ok(())
}
