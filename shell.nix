let
  sources = import ./nix;
  inherit (sources) pkgs;
in
pkgs.mkShell {
  name = "edgetpu";
  buildInputs = with pkgs; [
    cargo-edit
    cargo-udeps
    pkg-config
    tensorflow-lite
    meson
    # abseil-cpp
    flatbuffers
    libv4l
    boost
    xtensor
    opencv4
    v4l-utils
    rustToolchain
  ];
}
