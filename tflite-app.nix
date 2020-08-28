{ stdenv
, autoPatchelfHook
, tensorflow-lite
, flatbuffers
, libusb
, boost
, libedgetpu
, abseil-cpp
, enableDebugging
, xtensor
, xtensor-io
, opencv4
}:
stdenv.mkDerivation {
  pname = "tflite-app";
  version = "0.1.0";
  nativeBuildInputs = [ autoPatchelfHook stdenv.cc.cc.lib ];
  buildInputs = [
    libedgetpu.max
    libedgetpu.dev
    libedgetpu.basic.engine
    libedgetpu.basic.engine-native
    libedgetpu.posenet.decoder-op
    libedgetpu.basic.resource-manager
    libedgetpu.utils.error-reporter
    abseil-cpp
    tensorflow-lite
    flatbuffers
    boost
    xtensor
    opencv4
  ];
  dontConfigure = true;
  buildPhase = ''
    $CXX \
      -I ${opencv4}/include/opencv4 \
      -o tflite-app \
      -O3 \
      -flto \
      main.cpp \
      -std=c++2a \
      -lboost_program_options \
      -ledgetpu \
      -ledgetpu_basic_engine \
      -ledgetpu_basic_engine_native \
      -ltensorflow-lite \
      -lopencv_videoio \
      -lopencv_highgui \
      -lopencv_core \
      -lopencv_imgproc \
      -lrt \
      -lpthread \
      -ldl
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install tflite-app $out/bin
  '';
  src = ./app;
}
