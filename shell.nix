let
  pkgs = import ./nix;
  sources = import ./nix/sources.nix;
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
  buildInputs = with pkgs; [
    niv
    abseil-cpp
    cargo-edit
    cargo-udeps
    cargo-bloat
    clang_10
    flatbuffers
    glog
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
  ];

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
}
