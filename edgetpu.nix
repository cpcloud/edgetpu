{ stdenv
, bash
, gnumake
, coreutils
, zlib
, flatbuffers
, fetchurl
, dpkg
, autoPatchelfHook
, eigen
, fetchFromGitLab
, fetchFromGitHub
, libusb
, fetchzip
}:
let
  archs = { };
  mkLibEdgeTpu = { pname, sha256 }: stdenv.mkDerivation {
    inherit pname;
    version = "14.1";
    src = fetchurl {
      url = "https://packages.cloud.google.com/apt/pool/${pname}_14.1_arm64_${sha256}.deb";
      inherit sha256;
    };
    nativeBuildInputs = [ dpkg autoPatchelfHook ];
    buildInputs = [ libusb stdenv.cc.cc.lib ];
    unpackCmd = ''
      mkdir ./src
      dpkg -x $src ./src
    '';
    installPhase = ''
      # udev rules
      mkdir -p "$out/etc"
      cp -r ./lib/udev "$out/etc"

      # libs
      mkdir -p "$out/lib"
      chmod +x ./usr/lib/aarch64-linux-gnu/*
      cp -r ./usr/lib/aarch64-linux-gnu/* "$out/lib"

      # symlink libedgetpu.so -> libedgetpu.so.1 to allow linking via -ledgetpu
      ln -s $out/lib/libedgetpu.so{.1,}

      # docs
      mkdir -p "$out/share"
      cp -r ./usr/share/doc "$out/share"
    '';
  };

  libedgetpu1-max = mkLibEdgeTpu {
    pname = "libedgetpu1-max";
    sha256 = "795e7f49c81b1f9586f43b1978dd938b192df3e5e4939e1e8deb965d64ca41e6";
  };

  libedgetpu1-std = mkLibEdgeTpu {
    pname = "libedgetpu1-std";
    sha256 = "4669a44bd6d6f3b7d33c356182ceaeb29e1c981843629adeb77ac1283dbd498e";
  };

  libedgetpu-dev = stdenv.mkDerivation rec {
    pname = "libedgetpu-dev";
    version = "14.1";

    src = fetchurl {
      url = "https://packages.cloud.google.com/apt/pool/libedgetpu-dev_14.1_arm64_498e64beaac88b3de363dbf26fd20d98aa02db58d3e377945c7ed4127b8f139d.deb";
      sha256 = "498e64beaac88b3de363dbf26fd20d98aa02db58d3e377945c7ed4127b8f139d";
    };

    nativeBuildInputs = [ dpkg ];
    buildInputs = [ libedgetpu1-max ];
    unpackCmd = ''
      mkdir ./src
      dpkg -x $src ./src
      rm -rf ./src/usr/lib
    '';
    installPhase = ''
      # includes and docs
      mkdir -p "$out"
      cp -r ./usr/{include,share} "$out"
    '';
  };

  tflite-eigen = eigen.overrideAttrs (attrs: {
    version = "3.3.90";
    src = fetchFromGitLab {
      owner = "libeigen";
      repo = "eigen";
      rev = "3cd148f98338f8c03ce2757c528423338990a90d";
      sha256 = "0h4xzrkm3f3a692izw6v8r5sdm0yi4x3p4wqxvyp2sd0pnpcjz5f";
    };
  });

  gemmlowp-src = fetchzip {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip";
    sha256 = "1a27h0yay00ppjr7cm5lhpydd66a4c7phihhgiw1zsygaxkl2gy1";
  };

  neon-2-sse-src = fetchzip {
    url = "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/master.zip";
    sha256 = "1nwdgg286d91cd986z2h5d2is8k98bfpm4rh2iak59fd479xzp0x";
  };

  farmhash-src = fetchTarball {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz";
    sha256 = "1mqxsljq476n1hb8ilkrpb39yz3ip2hnc7rhzszz4sri8ma7qzp6";
  };

  fft2d-src = fetchTarball {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz";
    sha256 = "10f1qrk0n2pal6qndh2nk7vy6gvdbamc6myzbpqdp33c308gswvh";
  };

  tensorflow-lite = stdenv.mkDerivation rec {
    pname = "tensorflow-lite";
    version = "v2.0.0";
    src = fetchFromGitHub {
      owner = "tensorflow";
      repo = "tensorflow";
      rev = version;
      sha256 = "0zck3q6znmh0glak6qh2xzr25ycnhml7qcww7z8ynw2wbc75d7hp";
    };

    dontConfigure = true;
    buildPhase = ''
      substituteInPlace ./tensorflow/lite/tools/make/Makefile \
        --replace /bin/bash ${bash}/bin/bash \
        --replace /bin/sh ${bash}/bin/sh

      pushd ./tensorflow/lite/tools/make
      mkdir -p ./downloads/flatbuffers
      pushd ./downloads

      cp -r ${gemmlowp-src} ./gemmlowp
      cp -r ${neon-2-sse-src} ./neon_2_sse
      cp -r ${farmhash-src} ./farmhash
      # tflite is using the source of flatbuffers :(
      cp -r ${flatbuffers.src}/* ./flatbuffers
      cp -r ${fft2d-src} ./fft2d
      popd
      popd

      includes="-I. \
        -I./tensorflow/lite/tools/make \
        -I./tensorflow/lite/tools/make/downloads \
        -I./tensorflow/lite/tools/make/downloads/gemmlowp \
        -I./tensorflow/lite/tools/make/downloads/neon_2_sse \
        -I./tensorflow/lite/tools/make/downloads/farmhash/src \
        -I${tflite-eigen}/include/eigen3"
      ${gnumake}/bin/make \
        -j $(${coreutils}/bin/nproc) \
        -f ./tensorflow/lite/tools/make/Makefile \
        INCLUDES="$includes" \
        TARGET_TOOLCHAIN_PREFIX="" \
        all
    '';
    installPhase = ''
      mkdir "$out"

      # copy the static lib and binaries into the output dir
      cp -r ./tensorflow/lite/tools/make/gen/linux_aarch64/{bin,lib} "$out"

      # copy headers into the output dir
      find ./tensorflow/lite -name '*.h'| while read f; do
        install -D "$f" "$out/include/''${f/.\//}"
      done
    '';

    buildInputs = [
      zlib
      flatbuffers
    ];
  };

  tflite-app = stdenv.mkDerivation {
    pname = "tflite-app";
    version = "0.1.0";
    nativeBuildInputs = [ autoPatchelfHook ];
    buildInputs = [
      libedgetpu1-max
      libedgetpu-dev
      tensorflow-lite
      flatbuffers
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
  };
in
{
  inherit
    libedgetpu1-max
    libedgetpu1-std
    libedgetpu-dev
    tensorflow-lite
    tflite-app;
}
