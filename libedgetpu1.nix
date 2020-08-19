{ stdenv
, fetchurl
, dpkg
, autoPatchelfHook
, libusb
, kind
, arch
}:
let
  sha256s = {
    x86_64-linux = {
      max = "6f06f9f9e06aa960ef07e772339cd6d85672db7ea291b5cde37e0d7483414a1c";
      std = "c6cb84801d41bb06490d9ee18a0175c2a0b855a5d2865ae76e215a0ca2b9d1a4";
    };
    aarch64-linux = {
      max = "795e7f49c81b1f9586f43b1978dd938b192df3e5e4939e1e8deb965d64ca41e6";
      std = "4669a44bd6d6f3b7d33c356182ceaeb29e1c981843629adeb77ac1283dbd498e";
    };
  };
  system = stdenv.system;
  pname = "libedgetpu1-${kind}";
  sha256 = sha256s.${system}.${kind};
in
stdenv.mkDerivation {
  inherit pname;
  version = "14.1";
  src = fetchurl {
    url = "https://packages.cloud.google.com/apt/pool/${pname}_14.1_${arch}_${sha256}.deb";
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
    chmod +x ./usr/lib/${system}-gnu/*
    cp -r ./usr/lib/${system}-gnu/* "$out/lib"

    # symlink libedgetpu.so -> libedgetpu.so.1 to allow linking via -ledgetpu
    ln -s $out/lib/libedgetpu.so{.1,}

    # docs
    mkdir -p "$out/share"
    cp -r ./usr/share/doc "$out/share"
  '';
}
