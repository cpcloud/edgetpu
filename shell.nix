let
  sources = import ./nix;
  inherit (sources) pkgs;
in
pkgs.clang10Stdenv.mkDerivation {
  name = "edgetpu";
  buildInputs = with pkgs; [
    cargo-edit
    cargo-udeps
    clang_10
    pkg-config
    tensorflow-lite
    meson
    libedgetpu
    libcoral
    flatbuffers
    libv4l
    boost
    xtensor
    opencv4
    v4l-utils
    rustToolchain
  ];

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
  PROTOC = "${pkgs.protobuf}/bin/protoc";
}
