{ abseil-cpp
, boost
, cmake
, flatbuffers
, gobject-introspection
, libedgetpu
, meson
, opencv4
, pkg-config
, stdenv
, tensorflow-lite
, xtensor
, ninja
, cli11
}:
stdenv.mkDerivation {
  inherit "tflite-app";
  version = "1.0.0";

  nativeBuildInputs = [
    meson
    cmake
    pkg-config
    ninja
  ];

  buildInputs = [
    libedgetpu
    abseil-cpp
    cli11
    tensorflow-lite
    flatbuffers
    boost
    xtensor
    opencv4
  ];

  src = ./app;
}
