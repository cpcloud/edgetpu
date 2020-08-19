let
  pkgs = import <nixpkgs> { };
  rust = import ./rust.nix;
  edgetpu = import ./default.nix;
in
pkgs.mkShell {
  name = "edgetpu";
  nativeBuildInputs = [ pkgs.pkgconfig ];
  buildInputs = [
    edgetpu.tensorflow-lite
    edgetpu.libedgetpu1-max
    edgetpu.libedgetpu-dev
    pkgs.libusb
    pkgs.clang_10
    pkgs.openssl
    pkgs.sqlite
    pkgs.ffmpeg-full
    pkgs.libv4l
    pkgs.v4l-utils
    rust.nightly
    pkgs.flatbuffers
  ];

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
  PROTOC = "${pkgs.protobuf}/bin/protoc";
}
