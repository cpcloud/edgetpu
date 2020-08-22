let
  moz_overlay = import (
    builtins.fetchTarball https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz
  );
  pkgs = import (builtins.fetchTarball https://github.com/cpcloud/nixpkgs/archive/edgetpu.tar.gz) { overlays = [ moz_overlay ]; };
  extensions = [
    "clippy-preview"
    "rls-preview"
    "rustfmt-preview"
    "rust-analysis"
    "rust-std"
    "rust-src"
  ];
  channels = [
    { channel = "stable"; }
    { channel = "beta"; }
    { channel = "nightly"; date = "2020-07-12"; }
  ];
  environmentVariables = {
    LIBCLANG_PATH = "${pkgs.clang_10.cc.lib}/lib";
    CLANG_PATH = "${pkgs.clang_10}/bin/clang";
    PROTOC = "${pkgs.protobuf}/bin/protoc";
  };
  mkRust = { channel, date ? null }: with pkgs; {
    ${channel} = ((rustChannelOf { inherit channel date; }).rust.override { inherit extensions; });
  };
in
with pkgs;
lib.fold lib.mergeAttrs { } (map mkRust channels)
