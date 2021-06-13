{ abseil-cpp
, cmake
, flatbuffers
, gobject-introspection
, libedgetpu
, meson
, opencv4
, pkg-config
, glog
, stdenv
, tensorflow-lite
, xtensor
, ninja
, cli11
, libcoral
}:
stdenv.mkDerivation {
  pname = "tflite-app";
  version = "1.0.0";

  nativeBuildInputs = [
    meson
    cmake
    pkg-config
    ninja
  ];

  buildInputs = [
    libedgetpu
    libcoral
    abseil-cpp
    cli11
    glog
    tensorflow-lite
    flatbuffers
    xtensor
    opencv4
  ];

  src = ./app;
}
