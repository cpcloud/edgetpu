{ buildType ? "release" }:
let
  sources = import ./sources.nix;
in
import /home/cloud/src/nixpkgs {
  overlays = [
    (import ./opencv4.nix)
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
      glog = super.glog.overrideAttrs (attrs: {
        cmakeFlags = (attrs.cmakeFlags or []) ++ [ "-DCMAKE_CXX_STANDARD=17" ];
      });
      gflags = super.gflags.overrideAttrs (attrs: {
        cmakeFlags = (attrs.cmakeFlags or []) ++ [ "-DCMAKE_CXX_STANDARD=17" ];
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
      }).overrideAttrs (attrs: {
        dontStrip = buildType == "debug";
        NIX_CFLAGS_COMPILE = self.lib.optionalString
          (buildType == "debug")
          "${attrs.NIX_CFLAGS_COMPILE or ""} -ggdb -Og";
      });

      libedgetpu = (super.libedgetpu.override {
        inherit buildType;
        lto = buildType == "release";
      }).overrideAttrs (attrs: {
        dontStrip = buildType == "debug";
        NIX_CFLAGS_COMPILE = self.lib.optionalString
          (buildType == "debug")
          "${attrs.NIX_CFLAGS_COMPILE or ""} -ggdb -Og";
      });
    })
  ];
}
