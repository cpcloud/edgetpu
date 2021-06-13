use anyhow::Result;
use std::path::PathBuf;

const LIBS: &[&str] = &["opencv4", "edgetpu", "tensorflow-lite"];

fn main() -> Result<()> {
    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=static=tensorflow-lite");
    println!("cargo:rerun-if-changed=wrapper.h");

    // XXX: why tho?
    //
    // https://stackoverflow.com/questions/28060294/linking-to-a-c-library-that-has-extern-c-functions
    // Seems like using extern "C" functions isn't totally transparent.
    //
    // Is adding an attribute to all TfLite.* APIs with `#[link("tensorflow-lite")]` the real solution.
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
