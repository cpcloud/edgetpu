let
  sources = import ./nix/sources.nix;
  pkgs = import sources.nixpkgs {
    overlays = [
      (import ./nix/v4l-utils.nix)
      (import ./nix/xtensor.nix)
      (import ./nix/opencv4.nix)
    ];
  };
in
{
  tflite-app = pkgs.callPackage ./tflite-app.nix { };
}
