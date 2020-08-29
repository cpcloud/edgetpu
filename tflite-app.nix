{ llvmPackages_latest
, autoPatchelfHook
, tensorflow-lite
, flatbuffers
, boost
, libedgetpu
, abseil-cpp
, xtensor
, opencv4
, pkgconfig
, gst_all_1
, gobject-introspection
}:
llvmPackages_latest.stdenv.mkDerivation {
  pname = "tflite-app";
  version = "0.1.0";
  nativeBuildInputs = [
    autoPatchelfHook
    llvmPackages_latest.stdenv.cc.cc.lib
    pkgconfig
  ];
  buildInputs = [
    abseil-cpp
    tensorflow-lite
    flatbuffers
    boost
    xtensor
    opencv4
    gst_all_1.gst-plugins-bad
    gst_all_1.gst-plugins-base
    gst_all_1.gst-plugins-good
    gst_all_1.gst-plugins-ugly
    gst_all_1.gstreamer
    gobject-introspection
  ] ++ (
    with libedgetpu; [
      max
      dev
      basic.engine
      basic.engine-native
      posenet.decoder-op
      basic.resource-manager
      utils.error-reporter
    ]
  );
  dontConfigure = true;
  buildPhase = ''
    $CXX \
      -o tflite-app \
      -O3 \
      -flto \
      -std=c++17 \
      main.cpp \
      $(pkg-config --cflags opencv4) \
      $(pkg-config --libs opencv4) \
      -lboost_program_options \
      -ledgetpu \
      -ledgetpu_basic_engine \
      -ledgetpu_basic_engine_native \
      -ltensorflow-lite \
      -lrt \
      -lpthread \
      -ldl
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    install tflite-app $out/bin
  '';
  src = ./app;
  GST_DEBUG = 4;
}
