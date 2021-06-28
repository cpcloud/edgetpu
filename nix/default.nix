{ buildType ? "release" }:
let
  sources = import ./sources.nix;
in
import /home/cloud/src/nixpkgs {
  overlays = [
    (import ./opencv4.nix)
    (import ./v4l-utils.nix)
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
      glog = super.glog.overrideAttrs (old: {
        cmakeFlags = old.cmakeFlags ++ [ "-DCMAKE_CXX_STANDARD=17" ];
      });
      gflags = super.gflags.overrideAttrs (old: {
        cmakeFlags = old.cmakeFlags ++ [ "-DCMAKE_CXX_STANDARD=17" ];
      });
    })
    (self: super: {
      inherit (import sources.niv { }) niv;
    })
    (self: super: {
      libcoral = (super.libcoral.override {
        inherit buildType;
        withTests = [ ];
        lto = buildType == "release";
      }).overrideAttrs (_: {
        dontStrip = buildType == "debug";
      });

      libedgetpu = (super.libedgetpu.override {
        inherit buildType;
        lto = buildType == "release";
      }).overrideAttrs (_: {
        dontStrip = buildType == "debug";
      });
    })
  ];
}
