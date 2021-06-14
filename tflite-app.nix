{ abseil-cpp
, cmake
, flatbuffers
, libcoral
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
}:
stdenv.mkDerivation {
  pname = "tflite-app";
  version = "1.0.0";

  nativeBuildInputs = [
    meson
    cmake
    ninja
    pkg-config
  ];

  buildInputs = [
    abseil-cpp
    cli11
    flatbuffers
    glog
    libcoral
    libedgetpu
    opencv4
    tensorflow-lite
    xtensor
  ];

  src = ./app;
}
