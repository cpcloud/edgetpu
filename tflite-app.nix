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
    libedgetpu.posenet.decoder-tflite-plugin
    libedgetpu.utils.error-reporter
    abseil-cpp
    tensorflow-lite
    flatbuffers
    libusb
    boost
  ];
  dontConfigure = true;
  buildPhase = ''
    $CXX \
      -fPIC \
      -o tflite-app \
      -g \
      main.cpp \
      -ledgetpu \
      -ledgetpu_basic_engine \
      -ledgetpu_basic_engine_native \
      -ledgetpu_resource_manager \
      -ledgetpu_posenet_decoder_tflite_plugin \
      -ledgetpu_error_reporter \
      -lusb-1.0 \
      -lrt \
      -lpthread \
      -ldl \
      -Wl,--whole-archive \
      -ltensorflow-lite \
      -Wl,--no-whole-archive \
      -lboost_program_options
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install tflite-app $out/bin
  '';
  src = ./app;
}
