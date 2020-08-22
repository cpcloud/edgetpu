let
  pkgs = import (builtins.fetchTarball https://github.com/cpcloud/nixpkgs/archive/edgetpu.tar.gz) { };
  rust = import ./rust.nix;
in
pkgs.mkShell {
  name = "edgetpu";
  nativeBuildInputs = [ pkgs.pkgconfig ];
  buildInputs = [
    pkgs.tensorflow-lite
    pkgs.libedgetpu-max
    pkgs.libedgetpu-dev
    pkgs.libusb
    pkgs.clang_10
    pkgs.libv4l
    pkgs.v4l-utils
    rust.nightly
    pkgs.flatbuffers
    (pkgs.python3.withPackages (
      p: with p; [
        ipython
        numpy
        pillow
        edgetpu-max
        coloredlogs
      ]
    ))
  ];

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
  PROTOC = "${pkgs.protobuf}/bin/protoc";
}
