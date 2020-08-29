let
  raw-pkgs = builtins.fetchTarball https://github.com/cpcloud/nixpkgs/archive/2ee5cb241bb446c6282e285c415695cbe82f8508.tar.gz;
  pkgs = import raw-pkgs {
    overlays = [
      (self: super: {
        v4l-utils = super.v4l-utils.override {
          withGUI = false;
        };
      })
      (import ./xtensor.nix)
      # (import ./opencv4.nix)
    ];
  };
  guiLibs = pkgs.lib.optionals (!pkgs.stdenv.isAarch64) [
    pkgs.gtk3
  ];

in
{
  tflite-app = pkgs.callPackage ./tflite-app.nix { };
  shell = pkgs.mkShell
    {
      name = "edgetpu-shell";
      nativeBuildInputs = [ pkgs.pkgconfig ];
      buildInputs = guiLibs ++ (
        with pkgs;
        [
          tensorflow-lite
          libedgetpu.max
          libedgetpu.dev
          libedgetpu.basic.engine
          libedgetpu.basic.engine-native
          libedgetpu.posenet.decoder-op
          libedgetpu.basic.resource-manager
          libedgetpu.utils.error-reporter
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
      ) ++ pkgs.lib.optional (!pkgs.stdenv.isAarch64) pkgs.edgetpu-compiler;

      LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
      CLANG_PATH = "${pkgs.clang_10}/bin/clang";
      PROTOC = "${pkgs.protobuf}/bin/protoc";
    };
}
