let
  pkgs = import <nixpkgs> { };
  mkLibEdgeTpu = { pname, sha256 }: pkgs.stdenv.mkDerivation {
    inherit pname;
    version = "14.1";
    src = pkgs.fetchurl {
      url = "https://packages.cloud.google.com/apt/pool/${pname}_14.1_arm64_${sha256}.deb";
      inherit sha256;
    };
    nativeBuildInputs = [ pkgs.dpkg pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.libusb pkgs.stdenv.cc.cc.lib ];
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

  libedgetpu-dev = pkgs.stdenv.mkDerivation rec {
    pname = "libedgetpu-dev";
    version = "14.1";

    src = pkgs.fetchurl {
      url = "https://packages.cloud.google.com/apt/pool/libedgetpu-dev_14.1_arm64_498e64beaac88b3de363dbf26fd20d98aa02db58d3e377945c7ed4127b8f139d.deb";
      sha256 = "498e64beaac88b3de363dbf26fd20d98aa02db58d3e377945c7ed4127b8f139d";
    };

    nativeBuildInputs = [ pkgs.dpkg ];
    buildInputs = [ libedgetpu1-max ];
    unpackCmd = ''
      mkdir ./src
      dpkg -x $src ./src
      rm -rf ./src/usr/lib
    '';
    installPhase = ''
      # includes
      mkdir -p "$out"
      cp -r ./usr/include "$out"

      # docs
      mkdir -p "$out/share"
      cp -r ./usr/share/doc "$out/share"
    '';
  };

  eigen = pkgs.eigen.overrideAttrs (attrs: {
    version = "3.3.90";
    src = pkgs.fetchFromGitLab {
      owner = "libeigen";
      repo = "eigen";
      rev = "3cd148f98338f8c03ce2757c528423338990a90d";
      sha256 = "0h4xzrkm3f3a692izw6v8r5sdm0yi4x3p4wqxvyp2sd0pnpcjz5f";
    };
  });

  gemmlowp-src = pkgs.fetchurl {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip";
    sha256 = "0d3q9xnky50x8x8piwdrdg451a8mp3iw92lx4b9x1wi9v62b8y36";
  };

  neon-2-sse-src = pkgs.fetchurl {
    url = "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/master.zip";
    sha256 = "1l2cmqx7fl7hf78aig2sr285y0xahg512arcyhamgmvszr1kpssm";
  };

  farmhash-src = pkgs.fetchurl {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz";
    sha256 = "185b2xdxl4d4cnsnv6abg8s22gxvx8673jq2yaq85bz4cdy58q35";
  };

  fft2d-src = pkgs.fetchurl {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz";
    sha256 = "1jfflzi74fag9z4qmgwvp90aif4dpbr1657izmxlgvf4hy8fk9xd";
  };

  tensorflow-lite = pkgs.stdenv.mkDerivation rec {
    pname = "libtensorflow-lite";
    version = "v2.0.0";
    src = pkgs.fetchFromGitHub {
      owner = "tensorflow";
      repo = "tensorflow";
      rev = version;
      sha256 = "0zck3q6znmh0glak6qh2xzr25ycnhml7qcww7z8ynw2wbc75d7hp";
    };

    dontConfigure = true;
    buildPhase = ''
      substituteInPlace ./tensorflow/lite/tools/make/Makefile \
        --replace /bin/bash ${pkgs.bash}/bin/bash \
        --replace /bin/sh ${pkgs.bash}/bin/sh

      pushd ./tensorflow/lite/tools/make
      mkdir -p ./downloads/{gemmlowp,neon_2_sse,farmhash,fft2d,flatbuffers}
      pushd ./downloads

      unzip ${gemmlowp-src} -d .
      mv ./gemmlowp-*/* ./gemmlowp
      rm -rf ./gemmlowp-*

      unzip ${neon-2-sse-src} -d .
      mv ./ARM_NEON_2_x86_SSE-master/* ./neon_2_sse
      rm -rf ./ARM_NEON_2_x86_SSE-master

      tar -xzf ${farmhash-src} -C ./farmhash --strip-components=1

      # tflite is using the source of flatbuffers :(
      cp -r ${pkgs.flatbuffers.src}/* ./flatbuffers

      tar -xzf ${fft2d-src} -C ./fft2d --strip-components=1
      popd
      popd

      makedir=./tensorflow/lite/tools/make
      includes="-I$makedir -I.  \
        -I./tensorflow/lite/tools/make/downloads \
        -I./tensorflow/lite/tools/make/downloads/gemmlowp \
        -I./tensorflow/lite/tools/make/downloads/neon_2_sse \
        -I./tensorflow/lite/tools/make/downloads/farmhash/src \
        -I${eigen}/include/eigen3"
      make \
        -j $(nproc) \
        INCLUDES="$includes" \
        TARGET_TOOLCHAIN_PREFIX="" \
        -f ./tensorflow/lite/tools/make/Makefile \
        all
    '';
    installPhase = ''
      # make library dir
      mkdir -p "$out/lib" "$out/bin"

      # copy the static lib into the output dir
      install ./tensorflow/lite/tools/make/gen/linux_aarch64/lib/*.a "$out/lib"

      # copy binaries into the output dir
      install ./tensorflow/lite/tools/make/gen/linux_aarch64/bin/* "$out/bin"

      # copy headers into the output dir
      find ./tensorflow/lite -name '*.h'| while read f; do
        install -D "$f" "$out/include/''${f/.\//}"
      done
    '';
    postPatch = ''
      patchShebangs ./tensorflow/lite/tools/make
    '';

    buildInputs = [
      pkgs.gnumake
      pkgs.unzip
      pkgs.zlib
      pkgs.abseil-cpp
      pkgs.flatbuffers
    ];
  };

  tflite-app = pkgs.stdenv.mkDerivation {
    pname = "tflite-app";
    version = "0.1.0";
    nativeBuildInputs = [ pkgs.autoPatchelfHook ];
    buildInputs = [
      libedgetpu1-max
      libedgetpu-dev
      tensorflow-lite
      pkgs.flatbuffers
    ];
    dontConfigure = true;
    buildPhase = ''
      g++ \
        -o tflite_app main.cpp \
        -ledgetpu \
        -ltensorflow-lite \
        -lrt \
        -ldl
    '';
    installPhase = ''
      mkdir -p "$out/bin"
      install tflite_app $out/bin
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
