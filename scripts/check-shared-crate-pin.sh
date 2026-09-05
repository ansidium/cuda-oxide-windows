#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Keep shared host-crate sources identical in the workspace and every example.
# cargo oxide new reads these entries from the embedded workspace manifest.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
from pathlib import Path
import subprocess
import sys
import tomllib

shared = {"cuda-bindings", "cuda-core", "cuda-async"}
source_keys = {"version", "git", "rev", "tag", "branch", "registry", "path"}

def source_spec(dependency):
    if isinstance(dependency, str):
        return {"version": dependency}
    return {key: value for key, value in dependency.items() if key in source_keys}

def read_manifest(path):
    with Path(path).open("rb") as file:
        return tomllib.load(file)

root = read_manifest("Cargo.toml")["workspace"]["dependencies"]
expected = source_spec(root["cuda-core"])
errors = []
if "git" in expected and not any(key in expected for key in ("rev", "tag")):
    errors.append("Cargo.toml: shared git dependencies require a revision or tag")
for name in sorted(shared):
    if source_spec(root[name]) != expected:
        errors.append(f"Cargo.toml: {name} does not match cuda-core")

paths = subprocess.check_output(
    ["git", "ls-files", "crates/rustc-codegen-cuda/examples/**/Cargo.toml"],
    text=True,
).splitlines()
for path in paths:
    manifest = read_manifest(path)
    tables = [manifest, *manifest.get("target", {}).values()]
    for table in tables:
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name, dependency in table.get(section, {}).items():
                package = dependency.get("package", name) if isinstance(dependency, dict) else name
                if package in shared and source_spec(dependency) != expected:
                    errors.append(f"{path}: {name} has a different shared runtime source")

if errors:
    sys.exit("\n".join(f"error: {error}" for error in errors))
print(f"OK: shared host-crate sources agree in {len(paths)} example manifests: {expected}")
PY
