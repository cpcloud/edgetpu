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
, fmt
}:
(llvmPackages_latest.stdenv.mkDerivation {
  pname = "posecpp";
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
    fmt
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
    rm -f pose_src/*.o model-info_src/*.o
    mv pose model-info $out/bin
  '';
  src = ../pose;
})
