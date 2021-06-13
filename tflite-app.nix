{ abseil-cpp
, clang10Stdenv
, cmake
, flatbuffers
, gobject-introspection
, libedgetpu
, meson
, opencv4
, pkg-config
, glog
, tensorflow-lite
, xtensor
, ninja
, cli11
, libcoral
}:
clang10Stdenv.mkDerivation {
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
