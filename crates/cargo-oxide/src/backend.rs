/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend discovery and building.
//!
//! Finds or builds the `rustc_codegen_cuda` backend dynamic library using this priority:
//!
//! 1. `CUDA_OXIDE_BACKEND` env var (explicit override)
//! 2. Project config (`.cargo/cuda-oxide.toml`)
//! 3. Backend dynamic library next to the running `cargo-oxide` executable
//!    (packaged release zip layout)
//! 4. Local repo (detected by presence of `crates/rustc-codegen-cuda`)
//! 5. Cached backend at `~/.cargo/cuda-oxide/<platform filename>`,
//!    but only when it isn't older than the running `cargo-oxide` binary
//! 6. Auto-fetch from git and build (one-time, or after a stale-cache miss)
//!
//! ## Cache staleness (issue #49)
//!
//! `cargo install` always rewrites `~/.cargo/bin/cargo-oxide` on every
//! upgrade, bumping its mtime. The cached backend is only ever written by
//! step 6 below, so a binary newer than the cache is the canonical signal
//! that the user has just upgraded `cargo-oxide` and the cached backend
//! no longer matches the binary loading it. When step 5 detects that, we
//! drop both the cached backend *and* the cached source tree so that step 6
//! re-clones fresh and rebuilds, rather than rebuilding from a clone that
//! was taken whenever the user first installed.
//!
//! ## Cache staleness vs. source (backend source advances)
//!
//! The binary-mtime check above does not fire when the developer updates
//! the backend SOURCE (the `rustc-codegen-cuda` crate) but leaves the
//! `cargo-oxide` binary unchanged. In that case the cached `.so` is older
//! than the source it was built from, yet the binary check sees no upgrade
//! and the stale backend is silently reused. To catch this we also compare
//! the cached `.so` against the newest mtime of the backend source inputs
//! (the crate's `src/**` and `Cargo.toml`) found in the cached source tree.
//! When the source tree cannot be located we degrade gracefully to the
//! binary-only check rather than erroring.
//!
//! The two stale signals call for different recovery. A binary upgrade means
//! the cached source may no longer match the new binary, so we drop the
//! source tree and re-clone fresh (above). A source advance means the cached
//! source IS the newer truth, so we rebuild the `.so` from that existing
//! source in place; re-cloning would throw away the very source that
//! triggered the rebuild. Binary staleness takes precedence when both fire.
//!
//! ## Cache staleness vs. toolchain (the active rustc changes)
//!
//! The mtime checks above miss a toolchain swap: the cached `.so` is
//! dynamically linked against one specific `librustc_driver-<hash>.so`, but a
//! repo `rust-toolchain.toml` or a changed default nightly leaves the
//! `cargo-oxide` binary and the cached source untouched. The stale `.so` then
//! loads against the wrong driver and fails with a cryptic
//! `librustc_driver-<hash>.so: cannot open shared object file`. To catch this
//! we record the active toolchain fingerprint (`rustc -vV`) next to the cached
//! `.so` at build time and compare it on every lookup; a recorded fingerprint
//! that differs from the active toolchain forces a fresh re-clone and rebuild.
//! This check has the highest precedence, since a toolchain mismatch makes the
//! cached `.so` unloadable regardless of mtimes. A cache predating the
//! fingerprint file defers to the mtime checks (a `cargo-oxide` reinstall or
//! `rm -rf ~/.cargo/cuda-oxide` heals those).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

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
/// 2. Project config (`.cargo/cuda-oxide.toml`)
/// 3. Packaged backend next to the running `cargo-oxide` executable
/// 4. Local repo build (crates/rustc-codegen-cuda)
/// 5. Cached build at ~/.cargo/cuda-oxide/
/// 6. Auto-fetch + build from git
pub fn find_or_build_backend(workspace_root: &Path, configured_backend: Option<&Path>) -> PathBuf {
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

    // 2. Project config
    if let Some(path) = configured_backend {
        if path.exists() {
            return path.to_path_buf();
        }
        eprintln!(
            "Error: configured cuda-oxide backend does not exist: {}",
            path.display()
        );
        eprintln!("Build it or update `.cargo/cuda-oxide.toml`.");
        std::process::exit(1);
    }

    // 3. Packaged release layout: cargo-oxide.exe and rustc_codegen_cuda.dll
    // live side-by-side in the extracted archive. This keeps release users from
    // having to set CUDA_OXIDE_BACKEND manually.
    if let Some(packaged_backend) = packaged_backend_path(&backend_filename) {
        return packaged_backend;
    }

    // 4. Local repo
    let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
    if codegen_crate.is_dir() {
        let backend_path = backend_artifact_path(&codegen_crate, &host_target);
        build_backend_from_source(&codegen_crate);
        return backend_path;
    }

    // 5. Cached backend. Only honored when it isn't older than the running
    //    cargo-oxide binary; see the module-level comment about issue #49.
    if let Some(cache_dir) = cache_directory() {
        let cached_backend = cache_dir.join(&backend_filename);
        if cached_backend.exists() {
            let source_dir = cache_dir.join("src/crates/rustc-codegen-cuda");
            match cached_backend_status(&cached_backend, Some(&source_dir)) {
                CacheStatus::Fresh => return cached_backend,
                CacheStatus::StaleVsBinary => invalidate_cache(&cache_dir, &backend_filename),
                CacheStatus::StaleVsToolchain => {
                    eprintln!(
                        "Cached backend was built against a different Rust \
                         toolchain; re-cloning and rebuilding at {}.",
                        cache_dir.display()
                    );
                    invalidate_cache(&cache_dir, &backend_filename);
                }
                CacheStatus::StaleVsSource => {
                    // The cached source advanced; rebuild the backend from it
                    // place. We do NOT invalidate the cache here, so the
                    // auto-fetch step below skips the clone (the source tree is
                    // still present) and rebuilds from the existing source.
                    eprintln!(
                        "Cached backend source at {} is newer than the cached \
                         library; rebuilding from it in place.",
                        source_dir.display()
                    );
                }
            }
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
/// 2. Project config (`.cargo/cuda-oxide.toml`), returned even when missing
///    so the caller can report the configured-but-absent path.
/// 3. Packaged backend next to the running `cargo-oxide` executable.
/// 4. Local repo build path (`crates/rustc-codegen-cuda/target/<profile>/...`).
/// 5. Cache path at `~/.cargo/cuda-oxide/<platform filename>`.
///
/// `cargo oxide doctor` uses this so that a diagnostic run never triggers a
/// multi-minute backend build or a git clone before it can print anything.
pub fn backend_so_candidate(workspace_root: &Path, configured_backend: Option<&Path>) -> PathBuf {
    let host_target = active_host_target();
    let backend_filename = backend_filename_for_target(&host_target);

    if let Ok(path) = std::env::var("CUDA_OXIDE_BACKEND") {
        return PathBuf::from(path);
    }

    if let Some(path) = configured_backend {
        return path.to_path_buf();
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

/// Why the cached backend is out of date, or that it is current. The two
/// stale variants drive different recovery (re-clone vs. rebuild in place);
/// see the module-level comment.
#[derive(Debug, PartialEq, Eq)]
enum CacheStatus {
    /// Cache is up to date; reuse the cached `.so`.
    Fresh,
    /// The running `cargo-oxide` binary is newer than the cache: the user
    /// upgraded the binary, so the cached source may no longer match it.
    StaleVsBinary,
    /// The cached backend source is newer than the cached `.so`: the source
    /// was advanced in place and the `.so` should be rebuilt from it.
    StaleVsSource,
    /// The cached `.so` was built against a different Rust toolchain than the
    /// active one: it links a `librustc_driver` hash that no longer resolves,
    /// so it must be re-cloned and rebuilt. Highest precedence: an unloadable
    /// `.so` is stale regardless of mtimes.
    StaleVsToolchain,
}

/// Classifies the cached backend `.so` against the running `cargo-oxide`
/// binary (the user upgraded the binary) and the newest backend source input
/// (the developer advanced the source). When `source_dir` is `None`, or no
/// source inputs can be found under it, only the binary check applies.
/// Binary staleness takes precedence when both fire, since a binary upgrade
/// wants a fresh clone (which also picks up the newest source).
///
/// Conservative on errors: if we can't stat the cached `.so`, we report
/// [`CacheStatus::Fresh`] so a working cache is never invalidated on a failed
/// metadata read.
fn cached_backend_status(cached_so: &Path, source_dir: Option<&Path>) -> CacheStatus {
    let Ok(so_meta) = std::fs::metadata(cached_so) else {
        return CacheStatus::Fresh;
    };
    let Ok(so_mtime) = so_meta.modified() else {
        return CacheStatus::Fresh;
    };

    // Toolchain check (highest precedence): a toolchain swap makes the cached
    // `.so` unloadable no matter what the mtimes say, so it wins over the
    // binary/source mtime signals below.
    if let Some(cache_dir) = cached_so.parent()
        && toolchain_fingerprint_mismatch(cache_dir)
    {
        return CacheStatus::StaleVsToolchain;
    }

    let self_mtime = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok());

    // Binary check: if we can't stat our own executable, fall through to the
    // source check rather than declaring the cache fresh, so the source
    // signal is still honoured.
    if matches!(self_mtime, Some(self_mtime) if self_mtime > so_mtime) {
        return CacheStatus::StaleVsBinary;
    }

    let stale_vs_source = source_dir
        .and_then(newest_backend_source_mtime)
        .map(|src_mtime| src_mtime > so_mtime)
        .unwrap_or(false);
    if stale_vs_source {
        return CacheStatus::StaleVsSource;
    }

    CacheStatus::Fresh
}

/// File next to the cached `.so` recording the toolchain it was built against.
const TOOLCHAIN_FINGERPRINT_FILE: &str = "toolchain-fingerprint.txt";

/// A stable fingerprint of the active Rust toolchain: the full `rustc -vV`
/// output (release, commit-hash, host, LLVM version). The cached backend `.so`
/// links against this toolchain's `librustc_driver`, so any change here means
/// the cache can no longer be loaded.
fn current_toolchain_fingerprint() -> Option<String> {
    let output = Command::new("rustc").args(["-vV"]).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// True when the cached backend records a toolchain fingerprint that differs
/// from the active toolchain. Conservative: if the active fingerprint cannot be
/// read, or no fingerprint was recorded (a cache predating this check), returns
/// `false` and defers to the mtime checks rather than thrashing a working
/// cache. Pre-fingerprint caches are healed by the binary-mtime check on the
/// next `cargo-oxide` reinstall, or by `rm -rf ~/.cargo/cuda-oxide`.
fn toolchain_fingerprint_mismatch(cache_dir: &Path) -> bool {
    let Some(current) = current_toolchain_fingerprint() else {
        return false;
    };
    match std::fs::read_to_string(cache_dir.join(TOOLCHAIN_FINGERPRINT_FILE)) {
        Ok(stored) => stored.trim() != current,
        Err(_) => false,
    }
}

/// Records the active toolchain fingerprint next to the cached `.so`. Best
/// effort: a write failure just means the next run re-detects a mismatch and
/// rebuilds again.
fn write_toolchain_fingerprint(cache_dir: &Path) {
    if let Some(fp) = current_toolchain_fingerprint() {
        let _ = std::fs::write(cache_dir.join(TOOLCHAIN_FINGERPRINT_FILE), fp);
    }
}

/// Returns the newest mtime among the backend source inputs under
/// `source_dir`: every file in `src/**` plus the crate `Cargo.toml`.
///
/// Returns `None` when the directory cannot be located or yields no
/// readable inputs, which lets [`cached_backend_status`] degrade to the
/// binary-only check. The walk is best-effort: unreadable entries are
/// skipped rather than treated as failures.
fn newest_backend_source_mtime(source_dir: &Path) -> Option<SystemTime> {
    let mut newest: Option<SystemTime> = None;

    let mut consider = |path: &Path| {
        if let Ok(mtime) = std::fs::metadata(path).and_then(|m| m.modified()) {
            newest = Some(match newest {
                Some(cur) if cur >= mtime => cur,
                _ => mtime,
            });
        }
    };

    consider(&source_dir.join("Cargo.toml"));
    visit_files(&source_dir.join("src"), &mut consider);

    newest
}

/// Recursively visits every regular file under `dir`, calling `f` on each.
/// Best-effort: directories that cannot be read are skipped silently.
fn visit_files(dir: &Path, f: &mut dyn FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => visit_files(&path, f),
            Ok(ft) if ft.is_file() => f(&path),
            _ => {}
        }
    }
}

/// Drop both the cached `.so` and the cached source tree at `cache_dir`.
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
    let build_profile = backend_build_profile(&host_target);

    let mut cmd = Command::new("cargo");
    cmd.arg("build");
    if build_profile == "release" {
        cmd.arg("--release");
    }
    cmd.current_dir(codegen_crate);

    if let Some(ref path) = loader_path
        && !is_windows
        && let Some(value) = platform::append_env_paths("LIBRARY_PATH", vec![path.clone()])
    {
        cmd.env("LIBRARY_PATH", value);
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
        write_toolchain_fingerprint(&cache_dir);
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
        .join(backend_build_profile(target))
        .join(backend_filename_for_target(target))
}

fn backend_build_profile(target: &str) -> &'static str {
    if platform::is_windows_target(target) {
        "release"
    } else {
        "debug"
    }
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
    windows_vcpkg_roots_from_env(std::env::var_os("VCPKG_ROOT"), std::env::var_os("PATH"))
}

fn windows_vcpkg_roots_from_env(
    vcpkg_root: Option<OsString>,
    path: Option<OsString>,
) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root) = vcpkg_root.map(PathBuf::from).filter(|root| root.is_dir()) {
        push_unique_path(&mut roots, root);
    }

    if let Some(path) = path {
        for dir in std::env::split_paths(&path) {
            if vcpkg_executable_names()
                .iter()
                .any(|name| dir.join(name).is_file())
            {
                push_unique_path(&mut roots, dir);
            }
        }
    }

    roots
}

fn vcpkg_executable_names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["vcpkg.exe", "vcpkg"]
    } else {
        &["vcpkg"]
    }
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

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::StaleVsBinary,
            "cache backdated by 1y must be stale vs the running binary"
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

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::Fresh,
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
        assert_eq!(cached_backend_status(&so, None), CacheStatus::Fresh);
    }

    /// A backend source input newer than the cached `.so` must report
    /// `StaleVsSource` (the "developer advanced the source" case that issue
    /// #49's binary check alone misses). To isolate the source signal from
    /// the binary signal, the `.so` is future-dated past the running test
    /// binary, and the source file is dated later still.
    #[test]
    fn stale_when_source_postdates_cache() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Cache newer than the running binary so binary-staleness does NOT fire.
        let cache_mtime = SystemTime::now() + year;
        write_with_mtime(&so, b"built", cache_mtime);

        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        // Source newer than the cache: this is what trips source-staleness.
        write_with_mtime(
            &src.join("lib.rs"),
            b"// updated source",
            cache_mtime + year,
        );
        // Cargo.toml older than the .so; the src file is what trips staleness.
        write_with_mtime(&dir.join("Cargo.toml"), b"[package]", SystemTime::now());

        assert_eq!(
            cached_backend_status(&so, Some(&dir)),
            CacheStatus::StaleVsSource,
            "source newer than cached .so must be stale vs source (rebuild in place)"
        );
    }

    /// When every source input predates the cached `.so` (and the running
    /// binary too), the cache is fresh and must not be invalidated.
    #[test]
    fn fresh_when_source_predates_cache() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Cache far in the future so the running test binary can't make it stale.
        let cache_mtime = SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60);
        write_with_mtime(&so, b"built", cache_mtime);

        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_with_mtime(
            &src.join("lib.rs"),
            b"// old source",
            SystemTime::now() - Duration::from_secs(60),
        );
        write_with_mtime(
            &dir.join("Cargo.toml"),
            b"[package]",
            SystemTime::now() - Duration::from_secs(60),
        );

        assert_eq!(
            cached_backend_status(&so, Some(&dir)),
            CacheStatus::Fresh,
            "source older than cached .so must be reported fresh"
        );
    }

    /// A missing source tree must degrade to the binary-only check rather
    /// than erroring or spuriously invalidating a future-dated cache.
    #[test]
    fn fresh_when_source_dir_absent() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(
            &so,
            b"fresh",
            SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60),
        );
        let absent = dir.join("no-such-src-tree");
        assert_eq!(
            cached_backend_status(&so, Some(&absent)),
            CacheStatus::Fresh,
            "absent source tree must fall back to binary-only (fresh here)"
        );
    }

    /// When BOTH the running binary and the cached source postdate the `.so`,
    /// the binary signal wins so recovery re-clones fresh rather than
    /// rebuilding from a source tree that a binary upgrade may have outdated.
    #[test]
    fn binary_staleness_takes_precedence_over_source() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Backdate the `.so` so the freshly built test binary is newer than it.
        let base = SystemTime::now() - Duration::from_secs(365 * 24 * 60 * 60);
        write_with_mtime(&so, b"built", base);

        // Make the cached source newer than the `.so` too, so both signals fire.
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_with_mtime(
            &src.join("lib.rs"),
            b"// updated source",
            base + Duration::from_secs(30),
        );

        assert_eq!(
            cached_backend_status(&so, Some(&dir)),
            CacheStatus::StaleVsBinary,
            "binary staleness must win over source staleness"
        );
    }

    /// A cached `.so` whose recorded toolchain fingerprint differs from the
    /// active toolchain must be `StaleVsToolchain`, even when the mtimes alone
    /// would call it fresh. This is the case the mtime checks miss: the active
    /// rustc changed (e.g. a repo `rust-toolchain.toml`) while the binary and
    /// source are untouched, leaving the cached `.so` linked against a
    /// `librustc_driver` hash that no longer resolves.
    #[test]
    fn stale_when_toolchain_fingerprint_differs() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Future-date the `.so` so the binary/source mtime checks cannot fire.
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(
            dir.join(TOOLCHAIN_FINGERPRINT_FILE),
            "rustc 0.0.0 (deadbeef 1970-01-01)",
        )
        .unwrap();

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::StaleVsToolchain,
            "a recorded fingerprint differing from the active toolchain must be stale"
        );
    }

    /// A cached `.so` whose recorded fingerprint matches the active toolchain
    /// (with fresh mtimes) must be `Fresh`.
    #[test]
    fn fresh_when_toolchain_fingerprint_matches() {
        let Some(fp) = current_toolchain_fingerprint() else {
            return; // no rustc here; nothing to assert
        };
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(TOOLCHAIN_FINGERPRINT_FILE), fp).unwrap();

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::Fresh,
            "a matching fingerprint with fresh mtimes must be fresh"
        );
    }

    /// A missing fingerprint file (a cache predating this check) must defer to
    /// the mtime checks rather than forcing a rebuild, so existing caches are
    /// not thrashed. Here the future-dated `.so` is therefore `Fresh`.
    #[test]
    fn missing_toolchain_fingerprint_defers_to_mtime() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        // No fingerprint file written.
        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::Fresh,
            "absent fingerprint must defer to mtime checks (fresh here)"
        );
    }

    /// The toolchain check has the highest precedence: a differing fingerprint
    /// wins even when the cache is also stale-vs-binary, because an unloadable
    /// `.so` must be re-cloned regardless of why else it is stale.
    #[test]
    fn toolchain_staleness_takes_precedence_over_binary() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Backdate the `.so` so binary-staleness would otherwise fire.
        write_with_mtime(&so, b"built", SystemTime::now() - year);
        std::fs::write(
            dir.join(TOOLCHAIN_FINGERPRINT_FILE),
            "rustc 0.0.0 (deadbeef 1970-01-01)",
        )
        .unwrap();

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::StaleVsToolchain,
            "toolchain mismatch must win over binary staleness"
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
