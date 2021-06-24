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
      inherit (import sources.niv { }) niv;
    })
  ];
}
