let
  pkgs = import ./nix;
  sources = import ./nix/sources.nix;
  niv = (import sources.niv { }).niv;
  pythonEnv = pkgs.python3.withPackages(p: with p; [
    click
    ipdb
    ipython
    black
    mypy
    ujson
    opencv4
  ]);
in
pkgs.mkShell {
  name = "edgetpu";
  buildInputs = [ niv ] ++ (with pkgs; [
    abseil-cpp
    cargo-edit
    cargo-udeps
    cargo-bloat
    clang_10
    flatbuffers
    libcoral
    libedgetpu
    libv4l
    meson
    openblas
    opencv4
    pkg-config
    rustToolchain
    tensorflow-lite
    v4l-utils
    pythonEnv
  ]);

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
}
