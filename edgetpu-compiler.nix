{ stdenv
, fetchurl
, dpkg
, autoPatchelfHook
, libcxx
, libcxxabi
, lib
}:
let
  pname = "edgetpu-compiler";
  sha256 = "ef6eef29200270dcb941d2c1defa39c7d80e9c6f30cf7ced1c653a30bde0a502";
in
stdenv.mkDerivation {
  inherit pname;
  version = "14.1";
  src = fetchurl {
    url = "https://packages.cloud.google.com/apt/pool/${pname}_14.1_amd64_${sha256}.deb";
    inherit sha256;
  };
  nativeBuildInputs = [ dpkg autoPatchelfHook ];
  buildInputs = [ libcxx libcxxabi ];
  unpackCmd = ''
    mkdir ./src
    dpkg -x $src ./src
  '';
  installPhase = ''
    # binaries
    mkdir -p "$out/bin"
    chmod +x ./usr/bin/edgetpu_compiler_bin/edgetpu_compiler
    cp -r ./usr/bin/edgetpu_compiler_bin/edgetpu_compiler "$out/bin"

    # docs
    mkdir -p "$out/share"
    cp -r ./usr/share "$out"
  '';

  meta = {
    platforms = [ "x86_64-linux" ];
  };
}
