{ stdenv
, bash
, gnumake
, coreutils
, zlib
, flatbuffers
, eigen
, fetchFromGitLab
, fetchFromGitHub
, fetchzip
, arch
}:
let
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
in
stdenv.mkDerivation rec {
  pname = "tensorflow-lite";
  version = "v2.0.0";

  src = fetchFromGitHub {
    owner = "tensorflow";
    repo = "tensorflow";
    rev = version;
    sha256 = "0zck3q6znmh0glak6qh2xzr25ycnhml7qcww7z8ynw2wbc75d7hp";
  };

  buildInputs = [ zlib flatbuffers ];

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
    cp -r ${fft2d-src} ./fft2d

    # tflite is using the source of flatbuffers :(
    cp -r ${flatbuffers.src}/* ./flatbuffers

    popd

    popd

    includes="-I $PWD \
      -I $PWD/tensorflow/lite/tools/make/downloads/gemmlowp \
      -I $PWD/tensorflow/lite/tools/make/downloads/neon_2_sse \
      -I $PWD/tensorflow/lite/tools/make/downloads/farmhash/src \
      -I ${tflite-eigen}/include/eigen3"

    ${gnumake}/bin/make \
      -j $(${coreutils}/bin/nproc) \
      -f $PWD/tensorflow/lite/tools/make/Makefile \
      INCLUDES="$includes" \
      TARGET_TOOLCHAIN_PREFIX="" \
      all
  '';
  installPhase = ''
    mkdir "$out"

    # copy the static lib and binaries into the output dir
    cp -r ./tensorflow/lite/tools/make/gen/linux_${arch}/{bin,lib} "$out"

    # copy headers into the output dir
    find ./tensorflow/lite -name '*.h'| while read f; do
      install -D "$f" "$out/include/''${f/.\//}"
    done
  '';
}
