let
  pkgs = import /home/cloud/code/nix/nixpkgs { };
in
{
  tflite-app = pkgs.callPackage ./tflite-app.nix { };
}
