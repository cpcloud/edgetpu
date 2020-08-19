{ stdenv, fetchurl, dpkg, arch }:
let
  sha256s = {
    x86_64-linux = "2eb1100e364a203ce6b55f294676534c4c79aa4c9337f8e50e1ddaa07edd2ada";
    aarch64-linux = "498e64beaac88b3de363dbf26fd20d98aa02db58d3e377945c7ed4127b8f139d";
  };

  system = stdenv.system;
  sha256 = sha256s.${system};
in
stdenv.mkDerivation rec {
  pname = "libedgetpu-dev";
  version = "14.1";

  src = fetchurl {
    url = "https://packages.cloud.google.com/apt/pool/libedgetpu-dev_${version}_${arch}_${sha256}.deb";
    inherit sha256;
  };

  nativeBuildInputs = [ dpkg ];
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
}
