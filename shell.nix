let
  moz_overlay = import (
    builtins.fetchTarball https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz
  );
  pkgs = import (builtins.fetchTarball https://github.com/cpcloud/nixpkgs/archive/edgetpu.tar.gz) {
    overlays = [
      moz_overlay
      (self: super: {
        v4l-utils = super.v4l-utils.override {
          withGUI = false;
        };
      })
    ];
  };
  extensions = [
    "clippy-preview"
    "rls-preview"
    "rustfmt-preview"
    "rust-analysis"
    "rust-std"
    "rust-src"
  ];
  channels = [
    { channel = "stable"; }
    { channel = "beta"; }
    { channel = "nightly"; date = "2020-07-12"; }
  ];
  environmentVariables = {
    LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
    CLANG_PATH = "${pkgs.clang_10}/bin/clang";
    PROTOC = "${pkgs.protobuf}/bin/protoc";
  };
  mkRust = { channel, date ? null }: (
    pkgs.rustChannelOf { inherit channel date; }
  ).rust.override { inherit extensions; };

in
pkgs.mkShell {
  name = "edgetpu-shell";
  nativeBuildInputs = [ pkgs.pkgconfig ];
  buildInputs = [
    pkgs.edgetpu-compiler
    pkgs.tensorflow-lite
    pkgs.libedgetpu.max
    pkgs.libedgetpu.dev
    pkgs.libedgetpu.basic.engine
    pkgs.libedgetpu.basic.engine-native
    pkgs.libedgetpu.posenet.decoder-op
    pkgs.libedgetpu.basic.resource-manager
    pkgs.libedgetpu.utils.error-reporter
    pkgs.abseil-cpp
    pkgs.clang_10
    pkgs.libv4l
    pkgs.gtk3
    pkgs.gobject-introspection
    (
      (pkgs.gst_all_1.gst-plugins-bad.override {
        opencv4 = null;
        directfb = null;
      }).overrideAttrs (attrs: {
        mesonFlags = attrs.mesonFlags ++ [ "-Dopencv=disabled" "-Ddirectfb=disabled" ];
      })
    )
    pkgs.gst_all_1.gst-plugins-base
    pkgs.gst_all_1.gst-plugins-good
    pkgs.gst_all_1.gst-plugins-ugly
    pkgs.gst_all_1.gstreamer
    pkgs.v4l-utils
    pkgs.flatbuffers
    (mkRust {
      channel = "nightly";
      date = "2020-07-12";
    })
    (pkgs.python3.withPackages (
      p: with p; [
        p.ipython
        p.numpy
        p.pillow
        p.edgetpu-max
        p.coloredlogs
        p.svgwrite
        p.gst-python
      ]
    ))
  ];

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
  PROTOC = "${pkgs.protobuf}/bin/protoc";
}
