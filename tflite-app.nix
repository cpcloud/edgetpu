{ stdenv
, autoPatchelfHook
, libedgetpu1-max
, libedgetpu-dev
, tensorflow-lite
, flatbuffers
, libusb
, boost
}:
stdenv.mkDerivation {
  pname = "tflite-app";
  version = "0.1.0";
  nativeBuildInputs = [ autoPatchelfHook ];
  buildInputs = [
    libedgetpu1-max
    libedgetpu-dev
    tensorflow-lite
    flatbuffers
    libusb
    boost
  ];
  dontConfigure = true;
  buildPhase = ''
    g++ \
      -o tflite-app \
      main.cpp \
      -ledgetpu \
      -ltensorflow-lite \
      -lusb-1.0 \
      -lrt \
      -lpthread \
      -ldl -lboost_program_options
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install tflite-app $out/bin
  '';
  src = ./app;
}
