fn main() {
    cxx_build::bridge("src/main.rs")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-flto")
        .compile("pose-edge");

    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=edgetpu_basic_engine");
    println!("cargo:rustc-link-lib=edgetpu_basic_engine_native");
    println!("cargo:rustc-link-lib=edgetpu_resource_manager");
    println!("cargo:rustc-link-lib=tensorflow-lite");
    println!("cargo:rerun-if-changed=wrapper.h");
}
