#!/usr/bin/env sh
# Only shader development requires glslc and spirv-val. Cargo embeds the
# checked-in SPIR-V, so downstream builds do not need a Vulkan SDK.
set -eu
cd "$(dirname "$0")"
for shader in scan reduce take; do
    glslc --target-env=vulkan1.2 -O "$shader.comp" -o "$shader.spv"
    spirv-val --target-env vulkan1.2 "$shader.spv"
done
# Catch stale source/binary pairs in CI without requiring every platform to use
# the exact same glslc version (different versions can emit different SPIR-V).
sha256sum common.glsl scan.comp reduce.comp take.comp scan.spv reduce.spv take.spv > SHA256SUMS
