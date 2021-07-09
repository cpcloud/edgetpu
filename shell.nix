let
  buildType = "debug";
  pkgs = import ./nix { inherit buildType; };
  inherit (pkgs) lib;
  pythonEnv = pkgs.python3.withPackages (
    p: with p; [
      click
      ipdb
      ipython
      black
      mypy
      ujson
      opencv4
    ]
  );
  debugAdapter =
    if buildType == "debug" then
      pkgs.stdenvAdapters.keepDebugInfo
    else
      pkgs.lib.id;
in
(debugAdapter pkgs.stdenv).mkDerivation {
  name = "edgetpu";
  buildInputs = (with pkgs; [
    abseil-cpp
    cargo-bloat
    cargo-edit
    clang_11
    flatbuffers
    glog
    libcoral
    libedgetpu
    libv4l
    niv
    opencv4
    pkg-config
    pythonEnv
    rustToolchain
    tensorflow-lite
  ]) ++ lib.optionals pkgs.stdenv.isx86_64 [ pkgs.edgetpu-compiler ]
  ++ lib.optionals pkgs.stdenv.isAarch64 [ pkgs.v4l-utils ];

  LIBCLANG_PATH = "${pkgs.clang_12.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_12}/bin/clang";
}
