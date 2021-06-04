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
let
  pname = "tflite-app";
in
stdenv.mkDerivation {
  inherit pname;
  version = "1.0.0";
  nativeBuildInputs = [ meson ];
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
  # dontConfigure = true;
  # buildPhase = ''
  #   $CXX \
  #     -o ${pname} \
  #     -O3 \
  #     -std=c++17 \
  #     main.cpp \
  #     -pthread \
  #     $(pkg-config --cflags opencv4) \
  #     $(pkg-config --libs opencv4) \
  #     -lboost_program_options \
  #     -ledgetpu \
  #     -ledgetpu_basic_engine \
  #     -ledgetpu_basic_engine_native \
  #     -ltensorflow-lite \
  #     -lrt \
  #     -ldl
  # '';
  # installPhase = ''
  #   mkdir -p "$out/bin"
  #   install ${pname} $out/bin
  # '';

  src = ./app;
}
