let
  sources = import ./sources.nix;
in
import sources.nixpkgs {
  overlays = [
    (import ./opencv4.nix)
    (import ./v4l-utils.nix)
    (self: super: {
      naersk = self.callPackage sources.naersk { };
    })
    (import sources.fenix)
    (self: super: {
      inherit (self.fenix.latest)
        rustc
        cargo
        clippy-preview
        rustfmt-preview
        rust-analysis
        rust-analyzer-preview
        rust-std
        rust-src;
      rustToolchain = self.fenix.latest.withComponents [
        "rustc"
        "cargo"
        "clippy-preview"
        "rustfmt-preview"
        "rust-analysis"
        "rust-analyzer-preview"
        "rust-std"
        "rust-src"
      ];
    })
    (self: super: {
      tflite-pose = self.naersk.buildPackage {
        root = ../tflite-pose;
        nativeBuildInputs = [
          self.clang_10
          self.pkg-config
          self.cmake
        ];

        buildInputs = [
          self.openssl
          self.abseil-cpp
          self.tensorflow-lite
          self.libedgetpu
          self.libcoral
          self.flatbuffers
          self.libv4l
          self.opencv4
        ];

        LIBCLANG_PATH = "${self.clang_10.cc.lib}/lib";
        CLANG_PATH = "${self.clang_10}/bin/clang";
      };

      tflite-pose-image = self.dockerTools.buildLayeredImage {
        name = "tflite-pose";
        config = {
          Entrypoint = [ "${self.tflite-pose}/bin/tflite-pose" ];
          Env = [
            "SSL_CERT_FILE=${self.cacert}/etc/ssl/certs/ca-bundle.crt"
          ];
        };
      };
    })
  ];
}
