let
  pkgs = import <nixpkgs> { };
  archs = {
    x86_64-linux = "amd64";
    aarch64-linux = "arm64";
  };
  system = pkgs.stdenv.system;
  arch = archs.${system};
  edgetpu = pkgs.callPackage ./edgetpu.nix { inherit arch; };
  libedgetpu1-max = pkgs.callPackage ./libedgetpu1.nix {
    inherit arch;
    kind = "max";
  };
  libedgetpu1-std = pkgs.callPackage ./libedgetpu1.nix {
    inherit arch;
    kind = "std";
  };
  libedgetpu-dev = pkgs.callPackage ./libedgetpu-dev.nix { inherit arch; };
  edgetpu-compiler = pkgs.callPackage ./edgetpu-compiler.nix { };
  tensorflow-lite = pkgs.callPackage ./tensorflow-lite.nix { inherit arch; };
  tflite-app = pkgs.callPackage ./tflite-app.nix {
    inherit tensorflow-lite libedgetpu1-max libedgetpu-dev;
  };
in
{
  inherit
    libedgetpu1-max
    libedgetpu1-std
    libedgetpu-dev
    tensorflow-lite
    tflite-app
    edgetpu-compiler;
}
