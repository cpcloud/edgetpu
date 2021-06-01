let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {
    overlays = [
      (import ./nix/v4l-utils.nix)
      (import ./nix/xtensor.nix)
      (import ./nix/opencv4.nix)
    ];
  };
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
  nativeBuildInputs = with pkgs; [ pkgconfig ];
  buildInputs = guiPkgs ++ (
    with pkgs;
    [
      niv
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
