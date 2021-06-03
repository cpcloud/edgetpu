let
  sources = import ./nix;
  inherit (sources) pkgs;
in
{
  tflite-app = pkgs.callPackage ./tflite-app.nix { };
}
