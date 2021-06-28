use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;

const LIBS: &[&str] = &["opencv4", "edgetpu", "tensorflow-lite", "coral"];

fn main() -> Result<()> {
    println!("cargo:rustc-link-lib=coral");
    println!("cargo:rustc-link-lib=edgetpu");
    println!(
        "cargo:rustc-link-search=dylib=/nix/store/h08799k5yby8al44qv3hsk50v7gakq7r-glog-0.4.0"
    );
    println!("cargo:rustc-link-lib=dylib=glog");
    println!("cargo:rustc-link-lib=static=tensorflow-lite");

    println!("cargo:rerun-if-changed=wrapper.h");

    println!("cargo:rerun-if-changed=src/coral_ffi.rs");
    println!("cargo:rerun-if-changed=src/coral_ffi.cc");
    println!("cargo:rerun-if-changed=include/coral_ffi.h");

    // XXX: why tho?
    //
    // https://stackoverflow.com/questions/28060294/linking-to-a-c-library-that-has-extern-c-functions
    // Seems like using extern "C" functions isn't totally transparent.
    //
    // Is adding an attribute to all TfLite.* APIs with `#[link(name = "tensorflow-lite")]` the real solution?
    println!("cargo:rustc-flags=-l static=stdc++");

    let mut bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .clang_arg("-O3")
        .no_copy("(?i)mutex")
        .blocklist_type("_bindgen_ty_1")
        .rustfmt_bindings(true)
        .newtype_enum(".+")
        .derive_debug(true)
        .impl_debug(true)
        .size_t_is_usize(true)
        .opaque_type("max_align_t");

    let mut include_paths = vec![];

    for include_path in LIBS
        .iter()
        .flat_map(|lib| pkg_config::Config::new().probe(lib).unwrap().include_paths)
    {
        bindings = bindings.clang_arg(format!("-I{}", include_path.display()));
        include_paths.push(include_path);
    }

    bindings
        .generate()
        .map_err(|_| anyhow!("unable to generate bindings"))?
        .write_to_file(
            PathBuf::from(
                std::env::var("OUT_DIR").context("OUT_DIR environment variabe not defined")?,
            )
            .join("bindings.rs"),
        )
        .context("failed to write bindings to file")?;

    cxx_build::bridge("src/coral_ffi.rs")
        .includes(&include_paths)
        .opt_level(3)
        .file("src/coral_ffi.cc")
        .flag_if_supported("-std=c++17")
        .compile("tflite_coral");

    Ok(())
}
