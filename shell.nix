let
  pkgs = import ./nix;
  sources = import ./nix/sources.nix;
  niv = (import sources.niv { }).niv;
in
pkgs.mkShell {
  name = "edgetpu";
  buildInputs = [ niv ] ++ (with pkgs; [
    cargo-edit
    cargo-udeps
    clang_10
    abseil-cpp
    pkg-config
    tensorflow-lite
    meson
    libedgetpu
    libcoral
    flatbuffers
    libv4l
    xtensor
    opencv4
    v4l-utils
    rustToolchain
  ]);

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
  PROTOC = "${pkgs.protobuf}/bin/protoc";
}
