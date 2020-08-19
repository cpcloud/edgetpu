{ stdenv, autoPatchelfHook }:
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
  ];
  dontConfigure = true;
  buildPhase = ''
    g++ \
      -o tflite-app main.cpp \
      -ledgetpu \
      -ltensorflow-lite \
      -lrt \
      -ldl
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install tflite-app $out/bin
  '';
  src = ./app;
}
