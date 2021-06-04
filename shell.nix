let
  sources = import ./nix;
  inherit (sources) pkgs;
  guiPkgs = pkgs.lib.optionals (!pkgs.stdenv.isAarch64) [
    pkgs.gtk3
  ];
  libedgetpuPkgs = with pkgs.libedgetpu; [
    max
    dev
    basic.engine
    basic.engine-native
    posenet.decoder-op
    basic.resource-manager
    utils.error-reporter
  ];
in
pkgs.mkShell {
  name = "edgetpu-shell";
  nativeBuildInputs = with pkgs; [ pkg-config ];
  buildInputs = guiPkgs ++ (
    with pkgs;
    [
      cargo-edit
      cargo-udeps
      pkgconfig
      pkg-config
      tensorflow-lite
      clang_10
      abseil-cpp
      flatbuffers
      libv4l
      boost
      xtensor
      opencv4
      gst_all_1.gst-plugins-bad
      gst_all_1.gst-plugins-base
      gst_all_1.gst-plugins-good
      gst_all_1.gst-plugins-ugly
      gst_all_1.gstreamer
      gobject-introspection
      v4l-utils
      rustToolchain
      (python3.withPackages (
        p: [
          p.ipython
          p.numpy
          p.pillow
          p.edgetpu-max
          p.coloredlogs
          p.svgwrite
          p.gst-python
          p.opencv4
        ]
      ))
    ]
  )
    ++ pkgs.lib.optional (!pkgs.stdenv.isAarch64) pkgs.edgetpu-compiler
    ++ libedgetpuPkgs;

  LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
  CLANG_PATH = "${pkgs.clang_10}/bin/clang";
  PROTOC = "${pkgs.protobuf}/bin/protoc";
}
