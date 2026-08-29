#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify that the fork uses one stable Rust toolchain definition.

set -euo pipefail
export LC_ALL=C

cd "$(dirname "$0")/.."

ROOT_PIN=rust-toolchain.toml
NESTED_PIN=crates/rustc-codegen-cuda/rust-toolchain.toml
SCAFFOLD=crates/cargo-oxide/src/commands/scaffold.rs
DEVCONTAINER=.devcontainer/devcontainer.json

command -v git >/dev/null 2>&1

PYTHON=
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; assert sys.version_info >= (3, 11)' >/dev/null 2>&1; then
        PYTHON=$candidate
        break
    fi
done
if test -z "$PYTHON"; then
    echo "error: Python 3.11 or newer is required" >&2
    exit 1
fi

test -s "$ROOT_PIN"
test -s "$SCAFFOLD"
test -s "$DEVCONTAINER"

if test -e "$NESTED_PIN"; then
    echo "error: $NESTED_PIN overrides the root stable toolchain" >&2
    exit 1
fi

"$PYTHON" - "$ROOT_PIN" "$SCAFFOLD" "$DEVCONTAINER" <<'PY'
import glob
import json
import re
import subprocess
import sys
import tomllib

root_path, scaffold_path, devcontainer_path = sys.argv[1:4]


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


with open(root_path, "rb") as handle:
    toolchain = tomllib.load(handle).get("toolchain", {})

channel = toolchain.get("channel")
components = toolchain.get("components")
if channel != "stable":
    sys.exit(f"{root_path} must select stable, found {channel!r}")
if not isinstance(components, list) or not components:
    sys.exit(f"{root_path} must list the required components")

required = {"rust-src", "rustc-dev", "rust-analyzer", "rustfmt", "clippy", "llvm-tools"}
missing = sorted(required.difference(components))
if missing:
    sys.exit(f"{root_path} is missing components: {' '.join(missing)}")

scaffold = read(scaffold_path)
source = 'include_str!("../../../../rust-toolchain.toml")'
if source not in scaffold:
    sys.exit(f"{scaffold_path} must include the root toolchain file")
if re.search(r'const RUST_TOOLCHAIN_TOML: &str = r#"', scaffold):
    sys.exit(f"{scaffold_path} contains a duplicated toolchain definition")

try:
    devcontainer = json.loads(read(devcontainer_path))
except ValueError as error:
    sys.exit(f"{devcontainer_path} is not valid JSON: {error}")

rust_features = [
    value
    for key, value in devcontainer.get("features", {}).items()
    if key.split(":")[0] == "ghcr.io/devcontainers/features/rust"
]
if len(rust_features) != 1:
    sys.exit(f"{devcontainer_path} must define one Rust feature")
feature = rust_features[0]
if feature.get("version") != channel:
    sys.exit(
        f"{devcontainer_path} selects {feature.get('version')!r}, "
        f"but {root_path} selects {channel!r}"
    )
container_components = [
    name for name in feature.get("components", "").split(",") if name
]
unknown = sorted(set(container_components).difference(components))
if unknown:
    sys.exit(
        f"{devcontainer_path} lists components absent from {root_path}: "
        + " ".join(unknown)
    )

docs = sorted(glob.glob("cuda-oxide-book/**/*.md", recursive=True))
if len(docs) < 20:
    sys.exit(f"found only {len(docs)} book pages")

block_pattern = re.compile(r"\x60\x60\x60toml\n(.*?)\x60\x60\x60", re.S)
channel_pattern = re.compile(r'^\s*channel\s*=\s*"([^"]+)"', re.M)
components_pattern = re.compile(r"^\s*components\s*=\s*\[(.*?)\]", re.M | re.S)
quoted = 0
failures = []

for path in docs:
    for block in block_pattern.findall(read(path)):
        if "[toolchain]" not in block:
            continue
        quoted += 1
        quoted_channel = channel_pattern.search(block)
        quoted_components = components_pattern.search(block)
        if quoted_channel and quoted_channel.group(1) != channel:
            failures.append(f"{path} selects {quoted_channel.group(1)!r}")
        if quoted_components:
            names = re.findall(r'"([^"]+)"', quoted_components.group(1))
            if set(names) != set(components):
                failures.append(f"{path} lists different components")

if quoted == 0:
    sys.exit("no toolchain block found in the book")

markdown = [
    path
    for path in subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode().split("\0")
    if path
]
dated = re.compile(r"nightly-\d{4}-\d{2}-\d{2}")
for path in markdown:
    for number, line in enumerate(read(path).splitlines(), start=1):
        if dated.search(line):
            failures.append(f"{path}:{number} contains a dated nightly")

if failures:
    print("error: stable toolchain references are inconsistent", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    sys.exit(1)

print(
    f"OK: {root_path} selects stable with {len(components)} components; "
    f"the scaffold, devcontainer, and {quoted} book block(s) agree."
)
PY
