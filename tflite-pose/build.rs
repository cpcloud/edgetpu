use anyhow::Result;
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo:rustc-link-lib=tensorflow-lite");
    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rerun-if-changed=wrapper.h");

    bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_arg("-x")
        .clang_arg("c++")
        .enable_cxx_namespaces()
        .whitelist_type("tflite::.*")
        .whitelist_var("tflite::.*")
        .whitelist_function("tflite::.*")
        .opaque_type("std::.*")
        .opaque_type("flatbuffers::.*")
        .generate()
        .unwrap()
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))?;
    Ok(())
}
