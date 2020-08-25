{ stdenv
, autoPatchelfHook
, tensorflow-lite
, flatbuffers
, libusb
, boost
, libedgetpu
, abseil-cpp
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
    libedgetpu.basic.resource-manager
    libedgetpu.utils.error-reporter
    abseil-cpp
    tensorflow-lite
    flatbuffers
    boost
  ];
  dontConfigure = true;
  buildPhase = ''
    $CXX \
      -o tflite-app \
      -flto \
      -O3 \
      main.cpp \
      -ledgetpu \
      -ledgetpu_basic_engine \
      -ledgetpu_basic_engine_native \
      -lboost_program_options
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install tflite-app $out/bin
  '';
  src = ./app;
}
