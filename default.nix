with import <nixpkgs> { };
let
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

    nativeBuildInputs = [ dpkg autoPatchelfHook ];
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

  custom-eigen = fetchurl {
    url = "https://bitbucket.org/eigen/eigen/get/049af2f56331.tar.gz";
    sha256 = "0s7bskv4qip53khajp0lpc0sawaf1lwl004lrc13dbzcfg3rmmpk";
  };

  custom-gemmlowp = fetchurl {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/12fed0cd7cfcd9e169bf1925bc3a7a58725fdcc3.zip";
    sha256 = "0d3q9xnky50x8x8piwdrdg451a8mp3iw92lx4b9x1wi9v62b8y36";
  };

  custom-googletest = fetchurl {
    url = "https://github.com/google/googletest/archive/release-1.8.0.tar.gz";
    sha256 = "1n5p1m2m3fjrjdj752lf92f9wq3pl5cbsfrb49jqbg52ghkz99jq";
  };

  custom-absl = fetchurl {
    url = "https://github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz";
    sha256 = "0w6h4qsmg0w01f61s40ni644cykwrbmkpcq8pm743i7dm9mkzndc";
  };

  custom-neon-2-sse = fetchurl {
    url = "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/master.zip";
    sha256 = "1l2cmqx7fl7hf78aig2sr285y0xahg512arcyhamgmvszr1kpssm";
  };

  custom-farmhash = fetchurl {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz";
    sha256 = "185b2xdxl4d4cnsnv6abg8s22gxvx8673jq2yaq85bz4cdy58q35";
  };

  custom-flatbuffers = fetchurl {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v1.11.0.tar.gz";
    sha256 = "02x54gyk76fjryg2yl0r9yb278bypmpnaa3jnyqlakq989k2hjiz";
  };

  custom-fft2d = fetchurl {
    url = "https://storage.googleapis.com/mirror.tensorflow.org/www.kurims.kyoto-u.ac.jp/~ooura/fft2d.tgz";
    sha256 = "1jfflzi74fag9z4qmgwvp90aif4dpbr1657izmxlgvf4hy8fk9xd";
  };

  libtensorflow-lite = stdenv.mkDerivation rec {
    pname = "libtensorflow-lite";
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
        --replace /bin/bash ${runtimeShell} \
        --replace /bin/sh ${runtimeShell}

      mkdir -p ./tensorflow/lite/tools/make/downloads/{eigen,gemmlowp,googletest,absl,neon_2_sse,farmhash,flatbuffers,fft2d}

      pushd ./tensorflow/lite/tools/make/downloads
      tar -xzf ${custom-eigen} -C ./eigen --strip-components=1

      unzip ${custom-gemmlowp} -d .
      mv ./gemmlowp-*/* ./gemmlowp
      rm -rf ./gemmlowp-*

      tar -xzf ${custom-googletest} -C ./googletest --strip-components=1
      tar -xzf ${custom-absl} -C ./absl --strip-components=1

      unzip ${custom-neon-2-sse} -d .
      mv ./ARM_NEON_2_x86_SSE-master/* ./neon_2_sse
      rm -rf ./ARM_NEON_2_x86_SSE-master

      tar -xzf ${custom-farmhash} -C ./farmhash --strip-components=1
      tar -xzf ${custom-flatbuffers} -C ./flatbuffers --strip-components=1
      tar -xzf ${custom-fft2d} -C ./fft2d --strip-components=1

      popd

      set -x

      make -j $(nproc) \
        -C . \
        -f ./tensorflow/lite/tools/make/Makefile \
        TARGET=aarch64 \
        TARGET_TOOLCHAIN_PREFIX="" \
        lib
    '';
    installPhase = ''
      mkdir -p "$out/lib" "$out/include"
      cp ./tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a "$out/lib"
      find ./tensorflow/lite -name '*.h'| while read f; do
        dir=$(dirname "$f")
        include_dir="$out/include/$dir"
        mkdir -p "$include_dir"
        cp "$f" "$include_dir"
      done
    '';
    postPatch = ''
      patchShebangs ./tensorflow/lite/tools/make
    '';

    buildInputs = [
      gnumake
      unzip
      which
    ];
  };

  app = stdenv.mkDerivation {
    pname = "tflite-test";
    version = "0.1.0";
    nativeBuildInputs = [ autoPatchelfHook ];
    buildInputs = [
      abseil-cpp
      libedgetpu1-max
      libedgetpu-dev
      flatbuffers
      libtensorflow-lite
    ];
    dontConfigure = true;
    buildPhase = ''
      g++ \
        -o tflite_app main.cpp \
        -ledgetpu \
        -ltensorflow-lite
    '';
    installPhase = ''
      mkdir -p "$out/bin"
      cp tflite_app "$out/bin"
    '';
    src = ./app;
  };

in
{
  inherit
    libedgetpu1-max
    libedgetpu1-std
    libedgetpu-dev
    libtensorflow-lite
    # app
    ;
}
