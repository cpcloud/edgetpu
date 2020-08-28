#!/usr/bin/env zsh

set -euo pipefail

nix build -f. tflite-app

not_edge=0
edge=0
for model in $(nix run nixpkgs.fd -c fd '\.tflite$' ../edgetpu-models/test_data | sort -u); do
  if [[ "${model}" =~ .*edgetpu\.tflite ]]; then
    ((++edge))
  else
    ((++not_edge))
    echo $model
    return 0
  fi

  if ./result/bin/tflite-app --path "${model}" 2>/dev/null; then
    ((++success))
  fi
done
echo "    edge models: ${edge}"
echo "not edge models: ${not_edge}"
echo "         models: $((edge + not_edge))"
