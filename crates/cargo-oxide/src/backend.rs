/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend discovery and building.
//!
//! Finds or builds the `rustc_codegen_cuda` backend dynamic library using this priority:
//!
//! 1. `CUDA_OXIDE_BACKEND` env var (explicit override)
//! 2. Backend dynamic library next to the running `cargo-oxide` executable
//!    (packaged release zip layout)
//! 3. Local repo (detected by presence of `crates/rustc-codegen-cuda`)
//! 4. Cached backend at `~/.cargo/cuda-oxide/<platform filename>`,
//!    but only when it isn't older than the running `cargo-oxide` binary
//! 5. Auto-fetch from git and build (one-time, or after a stale-cache miss)
//!
//! ## Cache staleness (issue #49)
//!
//! `cargo install` always rewrites `~/.cargo/bin/cargo-oxide` on every
//! upgrade, bumping its mtime. The cached backend is only ever written by
//! step 4 below, so a binary newer than the cache is the canonical signal
//! that the user has just upgraded `cargo-oxide` and the cached backend
//! no longer matches the binary loading it. When step 3 detects that, we
//! drop both the cached backend *and* the cached source tree so that step 4
//! re-clones fresh and rebuilds, rather than rebuilding from a clone that
//! was taken whenever the user first installed.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::platform;

const BACKEND_CRATE_NAME: &str = "rustc_codegen_cuda";

/// Finds the workspace root by walking up from CWD looking for Cargo.toml
/// with a `crates/rustc-codegen-cuda` directory.
pub fn find_workspace_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join("crates/rustc-codegen-cuda").is_dir() && dir.join("Cargo.toml").is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Returns the path to the codegen backend dynamic library, building it if necessary.
///
/// Discovery order:
/// 1. `CUDA_OXIDE_BACKEND` env var
/// 2. Packaged backend next to the running `cargo-oxide` executable
/// 3. Local repo build (crates/rustc-codegen-cuda)
/// 4. Cached build at ~/.cargo/cuda-oxide/
/// 5. Auto-fetch + build from git
pub fn find_or_build_backend(workspace_root: &Path) -> PathBuf {
    let host_target = active_host_target();
    let backend_filename = backend_filename_for_target(&host_target);

    // 1. Explicit override
    if let Ok(path) = std::env::var("CUDA_OXIDE_BACKEND") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return p;
        }
        eprintln!(
            "Warning: CUDA_OXIDE_BACKEND={} does not exist, falling back to auto-detection",
            path
        );
    }

    // 2. Packaged release layout: cargo-oxide.exe and rustc_codegen_cuda.dll
    // live side-by-side in the extracted archive. This keeps release users from
    // having to set CUDA_OXIDE_BACKEND manually.
    if let Some(packaged_backend) = packaged_backend_path(&backend_filename) {
        return packaged_backend;
    }

    // 3. Local repo
    let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
    if codegen_crate.is_dir() {
        let backend_path = backend_artifact_path(&codegen_crate, &host_target);
        build_backend_from_source(&codegen_crate);
        return backend_path;
    }

    // 4. Cached backend. Only honored when it isn't older than the running
    //    cargo-oxide binary; see the module-level comment about issue #49.
    if let Some(cache_dir) = cache_directory() {
        let cached_backend = cache_dir.join(&backend_filename);
        if cached_backend.exists() {
            if !cached_backend_is_stale(&cached_backend) {
                return cached_backend;
            }
            invalidate_cache(&cache_dir, &backend_filename);
        }
    }

    // 5. Auto-fetch from git
    auto_fetch_and_build()
}

/// Returns where the backend dynamic library lives (or would live), with NO side
/// effects: never builds, never clones, never touches the network.
///
/// Mirrors the discovery order of [`find_or_build_backend`] minus its
/// build/clone steps:
///
/// 1. `CUDA_OXIDE_BACKEND` env var, returned even when the file is missing
///    so the caller can report the configured-but-absent path.
/// 2. Packaged backend next to the running `cargo-oxide` executable.
/// 3. Local repo build path (`crates/rustc-codegen-cuda/target/debug/...`).
/// 4. Cache path at `~/.cargo/cuda-oxide/<platform filename>`.
///
/// `cargo oxide doctor` uses this so that a diagnostic run never triggers a
/// multi-minute backend build or a git clone before it can print anything.
pub fn backend_candidate(workspace_root: &Path) -> PathBuf {
    let host_target = active_host_target();
    let backend_filename = backend_filename_for_target(&host_target);

    if let Ok(path) = std::env::var("CUDA_OXIDE_BACKEND") {
        return PathBuf::from(path);
    }

    if let Some(packaged_backend) = packaged_backend_path(&backend_filename) {
        return packaged_backend;
    }

    let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
    if codegen_crate.is_dir() {
        return backend_artifact_path(&codegen_crate, &host_target);
    }

    cache_directory()
        .map(|dir| dir.join(&backend_filename))
        .unwrap_or_else(|| PathBuf::from(backend_filename))
}

/// Returns true when the cached backend dynamic library is older than the running
/// `cargo-oxide` binary, which means the user has upgraded the binary
/// since the cache was last built.
///
/// Conservative on errors: if we can't resolve our own executable path or
/// stat either file, we report "not stale" so a working cache is never
/// invalidated on a failed metadata read.
fn cached_backend_is_stale(cached_backend: &Path) -> bool {
    let Ok(self_path) = std::env::current_exe() else {
        return false;
    };
    let Ok(self_meta) = std::fs::metadata(&self_path) else {
        return false;
    };
    let Ok(backend_meta) = std::fs::metadata(cached_backend) else {
        return false;
    };
    let (Ok(self_mtime), Ok(backend_mtime)) = (self_meta.modified(), backend_meta.modified())
    else {
        return false;
    };
    self_mtime > backend_mtime
}

/// Drop both the cached backend and the cached source tree at `cache_dir`.
///
/// Removing `src/` is what forces the auto-fetch step to re-clone instead
/// of rebuilding from a checkout that was taken at first-install time.
/// Both removals are best-effort; if either fails (e.g. permissions), we
/// fall through to step 4, which will fail loudly with a clear error.
fn invalidate_cache(cache_dir: &Path, backend_filename: &str) {
    eprintln!(
        "Detected upgraded cargo-oxide; refreshing cached backend at {} (issue #49).",
        cache_dir.display()
    );
    let _ = std::fs::remove_file(cache_dir.join(backend_filename));
    let _ = std::fs::remove_dir_all(cache_dir.join("src"));
}

/// Builds the backend from a local source tree.
pub fn build_backend_from_source(codegen_crate: &Path) {
    println!("Building rustc-codegen-cuda backend...");

    let host_target = active_host_target();
    let is_windows = platform::is_windows_target(&host_target);
    let rustc_sysroot = get_rustc_sysroot();
    let loader_path = rustc_sysroot
        .as_ref()
        .map(|s| rustc_sysroot_loader_dir(s, &host_target));
    let windows_libffi = is_windows.then(find_windows_libffi_paths).flatten();

    let mut cmd = Command::new("cargo");
    cmd.args(["build"]).current_dir(codegen_crate);

    if let Some(ref path) = loader_path
        && !is_windows
    {
        cmd.env("LIBRARY_PATH", path);
    }

    let loader_env = platform::loader_env_var(&host_target);
    if is_windows {
        let mut paths = Vec::new();
        if let Some(ref libffi) = windows_libffi {
            if let Some(value) = platform::prepend_env_paths("LIB", vec![libffi.lib_dir.clone()]) {
                cmd.env("LIB", value);
            }
            if let Some(ref bin_dir) = libffi.bin_dir {
                paths.push(bin_dir.clone());
            }
        } else {
            eprintln!(
                "Note: Windows backend builds require ffi.lib. Install `libffi:x64-windows` with vcpkg or set LIBFFI_LIB_DIR."
            );
        }
        if let Some(ref path) = loader_path {
            paths.push(path.clone());
        }
        if !paths.is_empty()
            && let Some(value) = platform::prepend_env_paths(loader_env, paths)
        {
            cmd.env(loader_env, value);
        }
    } else if let Some(ref path) = loader_path
        && let Some(value) = platform::append_env_paths(loader_env, vec![path.clone()])
    {
        cmd.env(loader_env, value);
    }

    let status = cmd.status().expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Failed to build rustc-codegen-cuda");
        std::process::exit(status.code().unwrap_or(1));
    }

    let backend_path = backend_artifact_path(codegen_crate, &host_target);
    if backend_path.exists() {
        println!("✓ Backend built: {}", backend_path.display());
    } else {
        eprintln!(
            "Warning: Expected backend not found at {}",
            backend_path.display()
        );
    }
}

/// Returns the cache directory for cuda-oxide artifacts: `~/.cargo/cuda-oxide/`.
fn cache_directory() -> Option<PathBuf> {
    dirs_path().map(|d| d.join("cuda-oxide"))
}

/// Resolves the Cargo home directory (`$CARGO_HOME` or `$HOME/.cargo`).
fn dirs_path() -> Option<PathBuf> {
    std::env::var("CARGO_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cargo"))
        })
}

/// Clones the cuda-oxide repo into the cache directory and builds the backend.
///
/// This is the last-resort discovery path for external users who don't have
/// the repo checked out locally. The clone is shallow (`--depth 1`) to keep
/// the download small.
fn auto_fetch_and_build() -> PathBuf {
    let host_target = active_host_target();
    let backend_filename = backend_filename_for_target(&host_target);
    let cache_dir = cache_directory().unwrap_or_else(|| {
        eprintln!("Error: Cannot determine cache directory.");
        eprintln!("Set CARGO_HOME or HOME environment variable.");
        std::process::exit(1);
    });

    let src_dir = cache_dir.join("src");
    let backend_path = cache_dir.join(&backend_filename);

    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

    if !src_dir.join("Cargo.toml").exists() {
        eprintln!("Backend not found. Fetching cuda-oxide source (one-time setup)...");
        eprintln!();
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://github.com/NVlabs/cuda-oxide.git",
                src_dir.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run git clone. Is git installed?");

        if !status.success() {
            eprintln!("Failed to clone cuda-oxide repository.");
            eprintln!(
                "You can manually set CUDA_OXIDE_BACKEND=/path/to/{}",
                backend_filename
            );
            std::process::exit(1);
        }
    }

    let codegen_crate = src_dir.join("crates/rustc-codegen-cuda");
    build_backend_from_source(&codegen_crate);

    let built_backend = backend_artifact_path(&codegen_crate, &host_target);
    if built_backend.exists() {
        std::fs::copy(&built_backend, &backend_path).expect("Failed to copy backend to cache");
        eprintln!("✓ Backend cached at {}", backend_path.display());
    }

    backend_path
}

/// Returns the active rustc sysroot path (e.g., `~/.rustup/toolchains/nightly-...`).
///
/// Used to locate `libstd`, `librustc_driver`, and other compiler libraries that
/// must be on the platform loader path when loading the codegen backend.
pub fn get_rustc_sysroot() -> Option<String> {
    let output = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

/// Returns the active rustc host target tuple, falling back to the std OS name.
pub(crate) fn active_host_target() -> String {
    rustc_host_target().unwrap_or_else(|| std::env::consts::OS.to_string())
}

fn rustc_host_target() -> Option<String> {
    let output = Command::new("rustc").arg("-vV").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(str::to_string))
}

fn packaged_backend_path(backend_filename: &str) -> Option<PathBuf> {
    let exe_path = std::env::current_exe().ok()?;
    let backend_path = backend_path_next_to_exe(&exe_path, backend_filename)?;
    backend_path.is_file().then_some(backend_path)
}

fn backend_path_next_to_exe(exe_path: &Path, backend_filename: &str) -> Option<PathBuf> {
    exe_path.parent().map(|dir| dir.join(backend_filename))
}

fn backend_filename_for_target(target: &str) -> String {
    platform::dylib_filename(BACKEND_CRATE_NAME, target)
}

fn backend_artifact_path(codegen_crate: &Path, target: &str) -> PathBuf {
    codegen_crate
        .join("target")
        .join("debug")
        .join(backend_filename_for_target(target))
}

fn rustc_sysroot_loader_dir(sysroot: &str, target: &str) -> PathBuf {
    if platform::is_windows_target(target) {
        PathBuf::from(sysroot).join("bin")
    } else {
        PathBuf::from(sysroot).join("lib")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct WindowsLibffiPaths {
    lib_dir: PathBuf,
    bin_dir: Option<PathBuf>,
}

pub(crate) fn windows_libffi_library_dir() -> Option<PathBuf> {
    find_windows_libffi_paths().map(|paths| paths.lib_dir)
}

pub(crate) fn windows_libffi_loader_dir() -> Option<PathBuf> {
    find_windows_libffi_paths().and_then(|paths| paths.bin_dir)
}

fn find_windows_libffi_paths() -> Option<WindowsLibffiPaths> {
    let explicit_bin_dir = std::env::var_os("LIBFFI_BIN_DIR")
        .map(PathBuf::from)
        .filter(|path| path.is_dir());

    if let Some(paths) = std::env::var_os("LIBFFI_LIB_DIR")
        .map(PathBuf::from)
        .and_then(|lib_dir| libffi_paths_from_lib_dir(lib_dir, explicit_bin_dir.clone()))
    {
        return Some(paths);
    }

    if let Some(lib_paths) = std::env::var_os("LIB") {
        for lib_dir in std::env::split_paths(&lib_paths) {
            if let Some(paths) = libffi_paths_from_lib_dir(lib_dir, explicit_bin_dir.clone()) {
                return Some(paths);
            }
        }
    }

    for root in windows_vcpkg_roots() {
        if let Some(paths) = libffi_paths_from_vcpkg_root(&root, explicit_bin_dir.clone()) {
            return Some(paths);
        }
    }

    None
}

fn windows_vcpkg_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root) = std::env::var_os("VCPKG_ROOT").map(PathBuf::from) {
        push_unique_path(&mut roots, root);
    }
    push_unique_path(&mut roots, PathBuf::from(r"C:\BuildData\tools\vcpkg"));
    push_unique_path(&mut roots, PathBuf::from(r"C:\vcpkg"));
    roots
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn libffi_paths_from_vcpkg_root(
    root: &Path,
    explicit_bin_dir: Option<PathBuf>,
) -> Option<WindowsLibffiPaths> {
    let installed = root.join("installed").join("x64-windows");
    libffi_paths_from_lib_dir(
        installed.join("lib"),
        explicit_bin_dir.or_else(|| {
            let bin_dir = installed.join("bin");
            bin_dir.is_dir().then_some(bin_dir)
        }),
    )
}

fn libffi_paths_from_lib_dir(
    lib_dir: PathBuf,
    explicit_bin_dir: Option<PathBuf>,
) -> Option<WindowsLibffiPaths> {
    lib_dir
        .join("ffi.lib")
        .is_file()
        .then(|| WindowsLibffiPaths {
            bin_dir: explicit_bin_dir.or_else(|| {
                lib_dir.parent().and_then(|parent| {
                    let bin_dir = parent.join("bin");
                    bin_dir.is_dir().then_some(bin_dir)
                })
            }),
            lib_dir,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::time::{Duration, SystemTime};

    /// A cached `.so` whose mtime predates the running test binary should
    /// be reported stale. The test binary is `current_exe()`, which was
    /// just rebuilt by `cargo test`, so its mtime is necessarily newer
    /// than a file we explicitly backdate.
    #[test]
    fn stale_when_cache_predates_running_binary() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(
            &so,
            b"stale",
            SystemTime::now() - Duration::from_secs(365 * 24 * 60 * 60),
        );

        assert!(
            cached_backend_is_stale(&so),
            "cache backdated by 1y must be reported stale"
        );
    }

    /// A cached `.so` written *after* the running binary is fresh and
    /// must not be reported stale, otherwise we'd thrash the cache on
    /// every invocation.
    #[test]
    fn fresh_when_cache_postdates_running_binary() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(
            &so,
            b"fresh",
            SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60),
        );

        assert!(
            !cached_backend_is_stale(&so),
            "cache postdating the test binary must be reported fresh"
        );
    }

    /// Missing cache file: we report not-stale and the caller's
    /// `cached_so.exists()` guard is what skips it. This keeps the
    /// helper conservative on stat failures.
    #[test]
    fn not_stale_when_cache_file_missing() {
        let dir = tempdir();
        let so = dir.join("does_not_exist.so");
        assert!(!cached_backend_is_stale(&so));
    }

    #[test]
    fn backend_artifact_path_uses_windows_dylib_name() {
        let root = Path::new(r"C:\Program Files\cuda oxide\crates\rustc-codegen-cuda");
        assert_eq!(
            backend_artifact_path(root, "x86_64-pc-windows-msvc"),
            root.join("target")
                .join("debug")
                .join("rustc_codegen_cuda.dll")
        );
    }

    #[test]
    fn backend_artifact_path_preserves_linux_dylib_name() {
        let root = Path::new("/opt/cuda-oxide/crates/rustc-codegen-cuda");
        assert_eq!(
            backend_artifact_path(root, "x86_64-unknown-linux-gnu"),
            root.join("target")
                .join("debug")
                .join("librustc_codegen_cuda.so")
        );
    }

    #[test]
    fn packaged_backend_path_is_next_to_running_exe() {
        let exe = Path::new("/opt/cuda-oxide/bin/cargo-oxide");
        assert_eq!(
            backend_path_next_to_exe(exe, "rustc_codegen_cuda.dll"),
            Some(PathBuf::from("/opt/cuda-oxide/bin/rustc_codegen_cuda.dll"))
        );
    }

    #[test]
    fn rustc_sysroot_loader_dir_is_target_aware() {
        assert_eq!(
            rustc_sysroot_loader_dir(r"C:\rustup\toolchains\nightly", "x86_64-pc-windows-msvc"),
            PathBuf::from(r"C:\rustup\toolchains\nightly").join("bin")
        );
        assert_eq!(
            rustc_sysroot_loader_dir("/rustup/toolchains/nightly", "x86_64-unknown-linux-gnu"),
            PathBuf::from("/rustup/toolchains/nightly").join("lib")
        );
    }

    #[test]
    fn libffi_paths_from_vcpkg_root_finds_import_library_and_dll_dir() {
        let dir = tempdir();
        let installed = dir.join("installed").join("x64-windows");
        let lib_dir = installed.join("lib");
        let bin_dir = installed.join("bin");
        std::fs::create_dir_all(&lib_dir).unwrap();
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::write(lib_dir.join("ffi.lib"), b"import lib").unwrap();
        std::fs::write(bin_dir.join("ffi-8.dll"), b"dll").unwrap();

        assert_eq!(
            libffi_paths_from_vcpkg_root(&dir, None),
            Some(WindowsLibffiPaths {
                lib_dir,
                bin_dir: Some(bin_dir)
            })
        );
    }

    #[test]
    fn libffi_paths_from_lib_dir_preserves_explicit_bin_dir() {
        let dir = tempdir();
        let lib_dir = dir.join("custom-lib");
        let bin_dir = dir.join("custom-bin");
        std::fs::create_dir_all(&lib_dir).unwrap();
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::write(lib_dir.join("ffi.lib"), b"import lib").unwrap();

        assert_eq!(
            libffi_paths_from_lib_dir(lib_dir.clone(), Some(bin_dir.clone())),
            Some(WindowsLibffiPaths {
                lib_dir,
                bin_dir: Some(bin_dir)
            })
        );
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "cargo-oxide-backend-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_with_mtime(path: &Path, contents: &[u8], mtime: SystemTime) {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .unwrap();
        f.write_all(contents).unwrap();
        f.set_modified(mtime).unwrap();
    }
}
