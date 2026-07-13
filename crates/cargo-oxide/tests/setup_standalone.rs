/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempProject(PathBuf);

impl TempProject {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "cargo-oxide-standalone-setup-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn standalone_setup_uses_resolved_backend_without_building_the_project() {
    let project = TempProject::new();
    fs::write(
        project.path().join("Cargo.toml"),
        "[package]\nname = \"setup-regression\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    let backend = project.path().join("resolved-backend.dll");
    fs::write(&backend, b"test backend marker").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_cargo-oxide"))
        .arg("setup")
        .current_dir(project.path())
        .env("CUDA_OXIDE_BACKEND", &backend)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "standalone setup failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("Backend is ready"),
        "setup did not report the resolved backend"
    );
    assert!(
        !project.path().join("target").exists(),
        "setup must not build the standalone project"
    );
    assert!(
        !project.path().join("Cargo.lock").exists(),
        "setup must not run Cargo in the standalone project"
    );
}
