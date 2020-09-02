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
, tinyformat
, enableDebugging
}:
(llvmPackages_latest.stdenv.mkDerivation {
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
    tinyformat
    opencv4
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
  dontStrip = true;
  buildPhase = ''
    make clean
    make -j $NIX_BUILD_CORES
  '';
  dontConfigure = true;
  installPhase = ''
    mkdir -p $out/bin
    rm -f *.o
    cp tflite-app $out/bin
  '';
  src = ./app;
})
