/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use anyhow::{Context, Result, bail};
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

pub fn canonical_text_bytes(bytes: &[u8]) -> Cow<'_, [u8]> {
    if !bytes.windows(2).any(|pair| pair == b"\r\n") {
        return Cow::Borrowed(bytes);
    }

    let mut normalized = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\r' && bytes.get(index + 1) == Some(&b'\n') {
            normalized.push(b'\n');
            index += 2;
        } else {
            normalized.push(bytes[index]);
            index += 1;
        }
    }
    Cow::Owned(normalized)
}

pub fn sha256_text_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(sha256_bytes(&canonical_text_bytes(&bytes)))
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse JSON {}", path.display()))
}

pub fn pretty_json<T: Serialize>(value: &T) -> Result<String> {
    let mut result = serde_json::to_string_pretty(value)?;
    result.push('\n');
    Ok(result)
}

pub fn write_if_changed(path: &Path, contents: &str) -> Result<bool> {
    if fs::read(path)
        .ok()
        .is_some_and(|bytes| canonical_text_bytes(&bytes).as_ref() == contents.as_bytes())
    {
        return Ok(false);
    }
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    let temporary = temporary_sibling(path);
    fs::write(&temporary, contents).with_context(|| format!("write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("replace {}", path.display()))?;
    Ok(true)
}

pub fn check_contents(path: &Path, expected: &str) -> Result<()> {
    let actual =
        fs::read(path).with_context(|| format!("generated file is missing: {}", path.display()))?;
    if canonical_text_bytes(&actual).as_ref() != expected.as_bytes() {
        bail!(
            "generated file is stale: {} (run `cargo run -p cuda-intrinsics-gen -- generate`)",
            path.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_text_bytes_normalizes_windows_line_endings() {
        assert_eq!(
            canonical_text_bytes(b"first\r\nsecond\r\n").as_ref(),
            b"first\nsecond\n"
        );
        assert_eq!(
            canonical_text_bytes(b"first\nsecond\n").as_ref(),
            b"first\nsecond\n"
        );
    }

    #[test]
    fn canonical_text_bytes_preserves_lone_carriage_returns() {
        assert_eq!(
            canonical_text_bytes(b"first\rsecond\n").as_ref(),
            b"first\rsecond\n"
        );
    }
}

pub fn rustfmt_source(source: &str) -> Result<String> {
    let rustfmt = std::env::var_os("RUSTFMT").unwrap_or_else(|| "rustfmt".into());
    let mut child = Command::new(&rustfmt)
        .args(["--emit", "stdout", "--edition", "2024"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .with_context(|| format!("start {}", PathBuf::from(&rustfmt).display()))?;
    child
        .stdin
        .take()
        .context("rustfmt stdin is unavailable")?
        .write_all(source.as_bytes())
        .context("write generated Rust to rustfmt")?;
    let output = child.wait_with_output().context("wait for rustfmt")?;
    ensure_success(output.status.success(), "rustfmt failed for generated Rust")?;
    String::from_utf8(output.stdout).context("rustfmt emitted non-UTF-8 output")
}

fn ensure_success(success: bool, message: &str) -> Result<()> {
    if !success {
        bail!("{message}");
    }
    Ok(())
}

fn temporary_sibling(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .expect("output file has a name")
        .to_os_string();
    name.push(".cuda-intrinsics-gen.tmp");
    path.with_file_name(name)
}
