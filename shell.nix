let
  pkgs = import <nixpkgs> { };
  edgetpu = pkgs.callPackage ./edgetpu.nix { };
  rust = import ./rust.nix;
in
pkgs.mkShell {
  name = "edgetpu";
  nativeBuildInputs = [ pkgs.pkgconfig ];
  buildInputs = [
    edgetpu.libedgetpu1-max
    edgetpu.libedgetpu1-std
    edgetpu.libedgetpu-dev
    edgetpu.tensorflow-lite

    pkgs.clang_10
    pkgs.openssl
    pkgs.sqlite
    pkgs.ffmpeg-full
    pkgs.libv4l
    pkgs.v4l-utils
    rust.nightly
    pkgs.cargo-edit
  ];

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
  PROTOC = "${pkgs.protobuf}/bin/protoc";
}
