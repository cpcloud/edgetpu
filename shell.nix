let
  pkgs = import ./nix;
  sources = import ./nix/sources.nix;
  niv = (import sources.niv { }).niv;
in
pkgs.mkShell {
  name = "edgetpu";
  buildInputs = [ niv ] ++ (with pkgs; [
    abseil-cpp
    cargo-edit
    cargo-udeps
    clang_10
    flatbuffers
    libcoral
    libedgetpu
    libv4l
    meson
    opencv4
    pkg-config
    rustToolchain
    tensorflow-lite
    v4l-utils
  ]);

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
}
