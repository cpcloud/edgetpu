let
  sources = import ./sources.nix;
in
sources // {
  pkgs = import sources.nixpkgs {
    overlays = [
      (import ./opencv4.nix)
      (import ./xtensor.nix)
      (import ./v4l-utils.nix)
      (self: super: {
        naersk = self.callPackage sources.naersk { };
      })
      (self: super: {
        fenix = import sources.fenix { };
      })
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
          buildInputs = [ self.openssl self.pkg-config ];
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
  };
}
