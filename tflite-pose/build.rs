fn main() {
    cxx_build::bridge("src/main.rs")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .compile("pose-edge");

    println!("cargo:rustc-link-lib=tensorflow-lite");
    println!("cargo:rustc-link-lib=edgetpu");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=usb-1.0");
    println!("cargo:rerun-if-changed=wrapper.h");
}
