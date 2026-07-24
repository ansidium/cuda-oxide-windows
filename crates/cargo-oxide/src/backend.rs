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
//! 6. Auto-fetch the pinned Windows-fork revision and build (one-time, or
//!    after a stale-cache miss)
//!
//! ## Cache staleness (issue #49)
//!
//! `cargo install` always rewrites `~/.cargo/bin/cargo-oxide` on every
//! upgrade, bumping its mtime. The cached backend is only ever written by
//! step 6 below, so a binary newer than the cache is the canonical signal
//! that the user has just upgraded `cargo-oxide` and the cached backend
//! no longer matches the binary loading it. When step 5 detects that, we
//! drop both the cached backend *and* the cached source tree so that step 6
//! re-fetches the embedded revision and rebuilds, rather than trusting a
//! checkout created by a different CLI revision.
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
//! source tree and re-fetch the pinned revision (above). A newer mtime within
//! an otherwise clean, exact checkout means the same pinned source should be
//! rebuilt in place. Binary staleness takes precedence when both fire.
//!
//! ## Cache staleness vs. toolchain (the active rustc changes)
//!
//! The mtime checks above miss a toolchain swap: the cached `.so` is
//! dynamically linked against one specific `librustc_driver-<hash>.so`, but a
//! repo `rust-toolchain.toml` or a changed stable compiler leaves the
//! `cargo-oxide` binary and the cached source untouched. The stale `.so` then
//! loads against the wrong driver and fails with a cryptic
//! `librustc_driver-<hash>.so: cannot open shared object file`. To catch this
//! we record the active toolchain fingerprint (`rustc -vV`) next to the cached
//! `.so` at build time and compare it on every lookup; a recorded fingerprint
//! that differs from the active toolchain forces a pinned-source re-fetch and rebuild.
//! This check has the highest precedence, since a toolchain mismatch makes the
//! cached `.so` unloadable regardless of mtimes. A cache predating the
//! fingerprint file defers to the mtime checks (a `cargo-oxide` reinstall or
//! `rm -rf ~/.cargo/cuda-oxide` heals those).
//!
//! ## Concurrent cache transactions
//!
//! Cache validation, invalidation, pinned fetch, backend build, publication,
//! and fingerprint publication run under one OS-backed exclusive file lock.
//! The lock file is never interpreted as state and is intentionally retained:
//! the OS releases the lock when its handle or process closes, so a crashed
//! writer cannot poison the cache. Every new holder double-checks the cache
//! after acquiring the lock before deciding whether to rebuild it.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::platform;

const BACKEND_CRATE_NAME: &str = "rustc_codegen_cuda";
const BACKEND_CACHE_LOCK_FILE: &str = ".backend-cache.lock";
const WINDOWS_MSVC_LINKER_ENV: &str = "CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER";
const WINDOWS_MSVC_LLD_LINKER: &str = "lld-link";

pub(crate) const PINNED_SOURCE_REPOSITORY: &str =
    "https://github.com/ansidium/cuda-oxide-windows.git";
// This source commit may intentionally precede the cargo-oxide CLI commit:
// embedding a commit's own SHA is impossible. It must nevertheless contain
// the complete backend and library migration for the selected compiler.
pub(crate) const PINNED_SOURCE_REVISION: &str = "9c9fd03c8d393b63be4f138329b7c1702a09f62e";

struct BackendCacheLock {
    file: std::fs::File,
}

impl BackendCacheLock {
    fn acquire(cache_dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(cache_dir).map_err(|error| {
            format!(
                "create backend cache directory {}: {error}",
                cache_dir.display()
            )
        })?;

        let lock_path = cache_dir.join(BACKEND_CACHE_LOCK_FILE);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|error| format!("open backend cache lock {}: {error}", lock_path.display()))?;

        match file.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                eprintln!(
                    "Another cargo-oxide process is preparing the backend cache; waiting for it."
                );
                file.lock().map_err(|error| {
                    format!("lock backend cache {}: {error}", lock_path.display())
                })?;
            }
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(format!(
                    "try to lock backend cache {}: {error}",
                    lock_path.display()
                ));
            }
        }

        Ok(Self { file })
    }
}

impl Drop for BackendCacheLock {
    fn drop(&mut self) {
        if let Err(error) = self.file.unlock() {
            // Closing the file immediately after Drop still releases an OS
            // lock. Report the explicit-unlock failure without proceeding
            // under an assumed lock or deleting the persistent lock file.
            eprintln!("Warning: failed to unlock the backend cache: {error}");
        }
    }
}

fn with_locked_backend_cache<T>(
    cache_dir: &Path,
    transaction: impl FnOnce(&Path) -> T,
) -> Result<T, String> {
    let _lock = BackendCacheLock::acquire(cache_dir)?;
    Ok(transaction(cache_dir))
}

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
        return build_backend_from_source(&codegen_crate);
    }

    let cache_dir = cache_directory().unwrap_or_else(|| {
        eprintln!("Error: Cannot determine cache directory.");
        eprintln!("Set CARGO_HOME or HOME environment variable.");
        std::process::exit(1);
    });

    with_locked_backend_cache(&cache_dir, |locked_cache_dir| {
        find_or_build_cached_backend(locked_cache_dir, &backend_filename)
    })
    .unwrap_or_else(|error| {
        eprintln!("Failed to lock the cuda-oxide backend cache: {error}");
        std::process::exit(1);
    })
}

/// Double-check and, when needed, rebuild the cache while its transaction lock
/// is held by [`find_or_build_backend`].
fn find_or_build_cached_backend(cache_dir: &Path, backend_filename: &str) -> PathBuf {
    let cached_backend = cache_dir.join(backend_filename);
    if cached_backend.exists() {
        let source_root = cache_dir.join("src");
        if !source_checkout_matches_revision(&source_root) {
            eprintln!(
                "Cached backend source does not match pinned revision {}; \
                 re-fetching it at {}.",
                PINNED_SOURCE_REVISION,
                cache_dir.display()
            );
            invalidate_cache(cache_dir, backend_filename);
        } else {
            let source_dir = source_root.join("crates/rustc-codegen-cuda");
            match cached_backend_status(&cached_backend, Some(&source_dir)) {
                CacheStatus::Fresh => return cached_backend,
                CacheStatus::StaleVsBinary => invalidate_cache(cache_dir, backend_filename),
                CacheStatus::StaleVsToolchain => {
                    eprintln!(
                        "Cached backend was built against a different Rust \
                         toolchain; re-fetching pinned source and rebuilding at {}.",
                        cache_dir.display()
                    );
                    invalidate_cache(cache_dir, backend_filename);
                }
                CacheStatus::StaleVsSource => {
                    // The pinned checkout is still the exact source of truth,
                    // so rebuild the library from it in place.
                    eprintln!(
                        "Cached backend source at {} is newer than the cached \
                         library; rebuilding from it in place.",
                        source_dir.display()
                    );
                }
            }
        }
    }

    auto_fetch_and_build(cache_dir, backend_filename)
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
/// 4. Local repo host build path
///    (`crates/rustc-codegen-cuda/target/<host>/<profile>/...`).
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
        return backend_so_path_candidate(&codegen_crate);
    }

    cache_directory()
        .map(|dir| dir.join(&backend_filename))
        .unwrap_or_else(|| PathBuf::from(backend_filename))
}

/// Why the cached backend is out of date, or that it is current. The two
/// stale variants drive different recovery (re-fetch vs. rebuild in place);
/// see the module-level comment.
#[derive(Debug, PartialEq, Eq)]
enum CacheStatus {
    /// Cache is up to date; reuse the cached `.so`.
    Fresh,
    /// The running `cargo-oxide` binary is newer than the cache: the user
    /// upgraded the binary, so the cached source may no longer match it.
    StaleVsBinary,
    /// The exact cached backend checkout has source mtimes newer than the
    /// cached `.so`, so the `.so` should be rebuilt from it.
    StaleVsSource,
    /// The cached `.so` was built against a different Rust toolchain than the
    /// active one: it links a `librustc_driver` hash that no longer resolves,
    /// so the pinned source must be re-fetched and rebuilt. Highest precedence:
    /// an unloadable
    /// `.so` is stale regardless of mtimes.
    StaleVsToolchain,
}

/// Classifies the cached backend `.so` against the running `cargo-oxide`
/// binary (the user upgraded the binary) and the newest backend source input
/// (the developer advanced the source). When `source_dir` is `None`, or no
/// source inputs can be found under it, only the binary check applies.
/// Binary staleness takes precedence when both fire, since a binary upgrade
/// requires a clean checkout of the embedded source revision.
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

/// Host target triple of the active rustc, as reported by `rustc -vV`.
fn active_host_triple() -> Option<String> {
    current_toolchain_fingerprint()?
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(str::to_owned))
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
/// Removing `src/` is what forces the auto-fetch step to re-fetch instead
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
pub fn build_backend_from_source(codegen_crate: &Path) -> PathBuf {
    println!("Building rustc-codegen-cuda backend...");

    let host_target = active_host_target();
    let rustc_sysroot = get_rustc_sysroot();
    let loader_path = rustc_sysroot
        .as_ref()
        .map(|s| rustc_sysroot_loader_dir(s, &host_target));

    let mut cmd = backend_build_command(codegen_crate, loader_path.as_deref(), &host_target);
    let output = cmd.output().unwrap_or_else(|error| {
        eprintln!("Failed to run cargo build for rustc-codegen-cuda: {error}");
        std::process::exit(1);
    });

    render_cargo_diagnostics(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }

    if !output.status.success() {
        eprintln!("Failed to build rustc-codegen-cuda");
        std::process::exit(output.status.code().unwrap_or(1));
    }

    let so_path =
        backend_artifact_from_cargo_output(codegen_crate, &output.stdout).unwrap_or_else(|error| {
            eprintln!("Backend build succeeded, but {error}");
            std::process::exit(1);
        });
    if !so_path.is_file() {
        eprintln!(
            "Backend build reported {}, but that file does not exist",
            so_path.display()
        );
        std::process::exit(1);
    }
    println!("✓ Backend built: {}", so_path.display());
    so_path
}

fn backend_build_command(
    codegen_crate: &Path,
    loader_path: Option<&Path>,
    host_target: &str,
) -> Command {
    let codegen_crate = absolute_path(codegen_crate);
    let target_dir = codegen_crate.join("target");
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--locked", "--lib"]);
    if backend_build_profile(host_target) == "release" {
        cmd.arg("--release");
    }
    cmd.args([
        "--target",
        "host-tuple",
        "--message-format=json-render-diagnostics",
        "--target-dir",
    ])
    .arg(&target_dir)
    .current_dir(&codegen_crate);

    // The backend is a host rustc plugin, not an application artifact. Keep it
    // out of an application's target directory and override both
    // CARGO_BUILD_TARGET and `[build] target`: an explicit `--target
    // host-tuple` makes Cargo compile the dylib for the running toolchain.
    cmd.env("CARGO_TARGET_DIR", &target_dir);
    cmd.env_remove("CARGO_BUILD_TARGET");

    if platform::is_windows_target(host_target) {
        prefer_windows_lld_linker_for_backend(&mut cmd);

        let mut loader_paths = Vec::new();
        if let Some(libffi) = find_windows_libffi_paths() {
            if let Some(value) = platform::prepend_env_paths("LIB", vec![libffi.lib_dir]) {
                cmd.env("LIB", value);
            }
            if let Some(bin_dir) = libffi.bin_dir {
                loader_paths.push(bin_dir);
            }
        } else {
            eprintln!(
                "Note: Windows backend builds require ffi.lib. Install `libffi:x64-windows` with vcpkg or set LIBFFI_LIB_DIR."
            );
        }
        if let Some(path) = loader_path {
            loader_paths.push(path.to_path_buf());
        }
        if let Some(value) =
            platform::prepend_env_paths(platform::loader_env_var(host_target), loader_paths)
        {
            cmd.env(platform::loader_env_var(host_target), value);
        }
    } else if let Some(path) = loader_path {
        if let Some(value) = platform::append_env_paths("LIBRARY_PATH", vec![path.to_path_buf()]) {
            cmd.env("LIBRARY_PATH", value);
        }
        if let Some(value) = platform::append_env_paths(
            platform::loader_env_var(host_target),
            vec![path.to_path_buf()],
        ) {
            cmd.env(platform::loader_env_var(host_target), value);
        }
    }

    cmd
}

fn backend_build_profile(host_target: &str) -> &'static str {
    if platform::is_windows_target(host_target) {
        // The Windows dylib links a large rustc plugin graph. Debug builds now
        // exceed MSVC/lld-link object/export limits; release keeps the backend
        // below those linker ceilings.
        "release"
    } else {
        "debug"
    }
}

fn prefer_windows_lld_linker_for_backend(cmd: &mut Command) {
    if std::env::var_os(WINDOWS_MSVC_LINKER_ENV).is_some() {
        return;
    }
    if windows_executable_on_path(WINDOWS_MSVC_LLD_LINKER) {
        cmd.env(WINDOWS_MSVC_LINKER_ENV, WINDOWS_MSVC_LLD_LINKER);
    }
}

fn windows_executable_on_path(name: &str) -> bool {
    std::env::var_os("PATH")
        .as_deref()
        .is_some_and(|paths| windows_executable_in_path(name, paths))
}

fn windows_executable_in_path(name: &str, paths: &OsStr) -> bool {
    let has_extension = Path::new(name).extension().is_some();
    std::env::split_paths(paths).any(|dir| {
        let candidate = dir.join(name);
        candidate.is_file() || (!has_extension && dir.join(format!("{name}.exe")).is_file())
    })
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

fn backend_dylib_filename() -> String {
    format!(
        "{}rustc_codegen_cuda{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    )
}

fn backend_so_path_candidate(codegen_crate: &Path) -> PathBuf {
    let target_dir = codegen_crate.join("target");
    let host_target = active_host_target();
    let profile = backend_build_profile(&host_target);
    let profile_dir = active_host_triple()
        .map(|host| target_dir.join(host).join(profile))
        .unwrap_or_else(|| target_dir.join(profile));
    profile_dir.join(backend_dylib_filename())
}

fn render_cargo_diagnostics(stdout: &[u8]) {
    for line in String::from_utf8_lossy(stdout).lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            if !line.is_empty() {
                println!("{line}");
            }
            continue;
        };
        if let Some(rendered) = message
            .get("message")
            .and_then(|message| message.get("rendered"))
            .and_then(|rendered| rendered.as_str())
        {
            eprint!("{rendered}");
        }
    }
}

/// Select the backend path Cargo reported for this exact manifest and dylib
/// target. There is deliberately no guessed-path fallback: a successful Cargo
/// exit without this artifact must fail instead of loading an older file left
/// in `target/debug` by a previous host build.
fn backend_artifact_from_cargo_output(
    codegen_crate: &Path,
    stdout: &[u8],
) -> Result<PathBuf, String> {
    let expected_manifest = codegen_crate
        .join("Cargo.toml")
        .canonicalize()
        .map_err(|error| format!("could not resolve backend manifest: {error}"))?;
    let expected_filename = backend_dylib_filename();
    let mut artifacts = Vec::new();

    for line in String::from_utf8_lossy(stdout).lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if message.get("reason").and_then(|reason| reason.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let manifest_matches = message
            .get("manifest_path")
            .and_then(|path| path.as_str())
            .and_then(|path| Path::new(path).canonicalize().ok())
            .is_some_and(|path| path == expected_manifest);
        let target_matches = message
            .get("target")
            .and_then(|target| target.get("name"))
            .and_then(|name| name.as_str())
            == Some("rustc_codegen_cuda")
            && message
                .get("target")
                .and_then(|target| target.get("kind"))
                .and_then(|kind| kind.as_array())
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("dylib")));
        if !manifest_matches || !target_matches {
            continue;
        }

        let Some(filenames) = message.get("filenames").and_then(|files| files.as_array()) else {
            continue;
        };
        for filename in filenames.iter().filter_map(|file| file.as_str()) {
            let path = PathBuf::from(filename);
            if path.file_name() == Some(std::ffi::OsStr::new(&expected_filename))
                && !artifacts.contains(&path)
            {
                artifacts.push(path);
            }
        }
    }

    match artifacts.as_slice() {
        [artifact] => Ok(artifact.clone()),
        [] => Err(format!(
            "Cargo reported no `{expected_filename}` artifact for rustc_codegen_cuda"
        )),
        _ => Err(format!(
            "Cargo reported multiple `{expected_filename}` artifacts for rustc_codegen_cuda"
        )),
    }
}

/// How the shared cache compares to a backend built in this checkout.
///
/// `doctor` reports the backend the current context resolves to, which inside
/// the repository is the local build. Projects outside the repository resolve
/// to the cache instead, so the two can disagree without either check failing.
#[derive(Debug, PartialEq, Eq)]
pub enum CacheReport {
    /// No cached backend. External projects will fetch and build on first use.
    Absent,
    /// The cache is at least as new as the local build.
    UpToDate,
    /// The cache predates the local build, so external projects would load an
    /// older backend than this checkout produces.
    OlderThanLocal,
}

/// Path of the cached backend, whether or not it exists.
///
/// Exposed so `doctor` can report the backend external projects resolve to,
/// which is not the one the in-repo context uses.
pub fn cached_backend_path() -> Option<PathBuf> {
    cache_directory().map(|dir| dir.join("librustc_codegen_cuda.so"))
}

/// Compares a cached backend against one built locally.
///
/// Ordering is by mtime, matching the staleness checks elsewhere in this
/// module. An unreadable mtime on either side reports [`CacheReport::UpToDate`]
/// rather than warning: `doctor` should not raise an alarm it cannot
/// substantiate.
pub fn compare_cache_to_local(cached_so: &Path, local_so: &Path) -> CacheReport {
    if !cached_so.exists() {
        return CacheReport::Absent;
    }
    if !local_so.exists() {
        // Nothing built here to be newer than the cache.
        return CacheReport::UpToDate;
    }

    let mtime = |path: &Path| std::fs::metadata(path).and_then(|m| m.modified()).ok();
    match (mtime(cached_so), mtime(local_so)) {
        (Some(cached), Some(local)) if cached < local => CacheReport::OlderThanLocal,
        _ => CacheReport::UpToDate,
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

/// Fetches the pinned cuda-oxide Windows-fork revision into the cache directory
/// and builds the backend.
///
/// This is the last-resort discovery path for external users who don't have
/// the repo checked out locally. Only the exact pinned commit is fetched at
/// depth one; a moving default branch never participates in the build.
fn auto_fetch_and_build(cache_dir: &Path, backend_filename: &str) -> PathBuf {
    let src_dir = cache_dir.join("src");
    let backend_path = cache_dir.join(backend_filename);

    if !source_checkout_matches_revision(&src_dir) {
        eprintln!(
            "Backend not found. Fetching cuda-oxide source revision {} (one-time setup)...",
            PINNED_SOURCE_REVISION
        );
        eprintln!();
        if let Err(error) =
            fetch_source_at_revision(PINNED_SOURCE_REPOSITORY, PINNED_SOURCE_REVISION, &src_dir)
        {
            eprintln!("Failed to fetch pinned cuda-oxide source: {error}");
            eprintln!(
                "You can manually set CUDA_OXIDE_BACKEND=/path/to/{}",
                backend_filename
            );
            std::process::exit(1);
        }
    }

    let codegen_crate = src_dir.join("crates/rustc-codegen-cuda");
    let built_backend = build_backend_from_source(&codegen_crate);
    if built_backend.exists() {
        install_backend_into(cache_dir, backend_filename, &built_backend)
            .expect("Failed to copy backend to cache");
        eprintln!("✓ Backend cached at {}", backend_path.display());
    }

    backend_path
}

fn is_full_git_revision(revision: &str) -> bool {
    revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn source_checkout_matches_revision(source_dir: &Path) -> bool {
    source_checkout_matches_revision_at(source_dir, PINNED_SOURCE_REVISION)
}

fn source_checkout_matches_revision_at(source_dir: &Path, revision: &str) -> bool {
    if !is_full_git_revision(revision) || !source_dir.join("Cargo.toml").is_file() {
        return false;
    }

    let head = git_stdout(source_dir, &["rev-parse", "HEAD"]);
    if !matches!(head.as_deref(), Some(value) if value.eq_ignore_ascii_case(revision)) {
        return false;
    }

    matches!(
        git_stdout(source_dir, &["status", "--porcelain=v1"]),
        Some(status) if status.is_empty()
    )
}

fn git_stdout(source_dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(source_dir)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn checked_git(command: &mut Command, action: &str) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|error| format!("{action}: failed to start git: {error}"))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let detail = if stderr.is_empty() { stdout } else { stderr };
    let detail_suffix = if detail.is_empty() {
        String::new()
    } else {
        format!(": {detail}")
    };
    Err(format!(
        "{action}: git exited with {}{detail_suffix}",
        output.status
    ))
}

fn fetch_source_at_revision(
    repository: &str,
    revision: &str,
    source_dir: &Path,
) -> Result<(), String> {
    if !is_full_git_revision(revision) {
        return Err(format!(
            "embedded source revision must be a full 40-character Git SHA, got `{revision}`"
        ));
    }

    let parent = source_dir
        .parent()
        .ok_or_else(|| format!("source path has no parent: {}", source_dir.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|error| format!("create cache directory {}: {error}", parent.display()))?;

    if source_dir.exists() {
        match std::fs::remove_dir_all(source_dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "remove stale source checkout {}: {error}",
                    source_dir.display()
                ));
            }
        }
    }

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging_dir = parent.join(format!("src.fetch-{}-{nonce}", std::process::id()));

    let result = (|| {
        let mut init = Command::new("git");
        init.args(["init", "--quiet"]).arg(&staging_dir);
        checked_git(&mut init, "initialize source cache")?;

        let mut add_remote = Command::new("git");
        add_remote
            .arg("-C")
            .arg(&staging_dir)
            .args(["remote", "add", "origin", repository]);
        checked_git(&mut add_remote, "configure source remote")?;

        let mut fetch = Command::new("git");
        fetch.arg("-C").arg(&staging_dir).args([
            "fetch",
            "--depth=1",
            "--no-tags",
            "origin",
            revision,
        ]);
        checked_git(&mut fetch, "fetch pinned source revision")?;

        let mut checkout = Command::new("git");
        checkout.arg("-C").arg(&staging_dir).args([
            "checkout",
            "--detach",
            "--quiet",
            "FETCH_HEAD",
        ]);
        checked_git(&mut checkout, "check out pinned source revision")?;

        if !source_checkout_matches_revision_at(&staging_dir, revision) {
            return Err(format!(
                "fetched checkout did not verify as clean revision {revision}"
            ));
        }

        if let Err(error) = std::fs::rename(&staging_dir, source_dir) {
            // A concurrent cargo-oxide process may have published the same
            // exact checkout first. Accept only that verified outcome.
            if source_checkout_matches_revision_at(source_dir, revision) {
                let _ = std::fs::remove_dir_all(&staging_dir);
            } else {
                return Err(format!(
                    "publish source checkout {} -> {}: {error}",
                    staging_dir.display(),
                    source_dir.display()
                ));
            }
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging_dir);
    }
    result
}

/// Copies a freshly built backend into `cache_dir` and records the toolchain
/// fingerprint beside it.
///
/// The fingerprint must be written whenever the backend is. A backend installed
/// without one falls back to the mtime checks, which cannot see a toolchain
/// swap, so the next lookup would load a backend linked against the wrong
/// `librustc_driver`.
///
/// Takes the directory explicitly so it can be exercised without touching
/// `CARGO_HOME`.
fn install_backend_into(
    cache_dir: &Path,
    backend_filename: &str,
    built_backend: &Path,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(cache_dir)?;
    let backend_path = cache_dir.join(backend_filename);
    let source_is_destination = built_backend == backend_path
        || match (built_backend.canonicalize(), backend_path.canonicalize()) {
            (Ok(source), Ok(destination)) => source == destination,
            _ => false,
        };
    if !source_is_destination {
        std::fs::copy(built_backend, &backend_path)?;
    }
    write_toolchain_fingerprint(cache_dir);
    Ok(backend_path)
}

/// Publishes a freshly built backend to the shared cache at
/// `~/.cargo/cuda-oxide/`.
///
/// That path is what step 5 of the discovery order resolves to, and it is the
/// only one a project outside this repository can reach: `find_workspace_root`
/// walks up from the current directory looking for `crates/rustc-codegen-cuda`
/// and finds nothing from an unrelated crate.
///
/// Returns `None` when the cache directory cannot be determined or the copy
/// fails. Callers treat this as best effort: a failure leaves the in-repo build
/// usable and costs external projects only a rebuild.
pub fn publish_to_cache(built_backend: &Path) -> Option<PathBuf> {
    let cache_dir = cache_directory()?;
    let backend_filename = backend_filename_for_target(&active_host_target());
    with_locked_backend_cache(&cache_dir, |locked_cache_dir| {
        install_backend_into(locked_cache_dir, &backend_filename, built_backend)
    })
    .ok()?
    .ok()
}

/// Returns the active rustc sysroot path.
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
    use std::ffi::OsStr;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::{Duration, SystemTime};

    /// The codegen backend is part of the cuda-oxide toolchain, not the
    /// application being debugged/sanitized. A user-supplied
    /// `CARGO_TARGET_DIR` for the application must therefore not change where
    /// cargo-oxide builds or looks for `librustc_codegen_cuda.so`.
    #[test]
    fn backend_build_command_isolates_target_dir_and_forces_the_host() {
        let root = tempdir();
        let codegen = root.join("codegen");
        let rustc_lib = root.join("rustc").join("lib");
        let target_dir = codegen.join("target");
        let command = backend_build_command(&codegen, Some(&rustc_lib), "x86_64-unknown-linux-gnu");

        let cargo_target_dir = command
            .get_envs()
            .find_map(|(key, value)| (key == OsStr::new("CARGO_TARGET_DIR")).then_some(value));
        let cargo_build_target = command
            .get_envs()
            .find_map(|(key, value)| (key == OsStr::new("CARGO_BUILD_TARGET")).then_some(value));
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let target_dir_arg = target_dir.to_string_lossy().into_owned();

        assert_eq!(cargo_target_dir.flatten(), Some(target_dir.as_os_str()));
        assert_eq!(cargo_build_target, Some(None));
        assert_eq!(command.get_current_dir(), Some(codegen.as_path()));
        assert!(
            args.windows(2)
                .any(|args| args == ["--target", "host-tuple"])
        );
        assert!(
            args.windows(2)
                .any(|args| args[0] == "--target-dir" && args[1] == target_dir_arg)
        );
        assert!(
            args.iter()
                .any(|arg| arg == "--message-format=json-render-diagnostics")
        );
        assert!(args.iter().any(|arg| arg == "--locked"));
        assert!(!args.iter().any(|arg| arg == "--release"));
    }

    #[test]
    fn full_git_revision_validation_rejects_floating_refs() {
        assert!(is_full_git_revision(
            "0123456789abcdef0123456789abcdef01234567"
        ));
        assert!(!is_full_git_revision("main"));
        assert!(!is_full_git_revision("0123456789abcdef"));
        assert!(!is_full_git_revision(
            "z123456789abcdef0123456789abcdef01234567"
        ));
    }

    #[test]
    fn concurrent_first_run_cache_transactions_build_once_after_double_check() {
        let cache_dir = tempdir();
        let ready_marker = cache_dir.join("backend-ready");
        let build_count = Arc::new(AtomicUsize::new(0));
        let (first_locked_tx, first_locked_rx) = mpsc::channel();
        let (release_first_tx, release_first_rx) = mpsc::channel();

        let first_cache_dir = cache_dir.clone();
        let first_ready_marker = ready_marker.clone();
        let first_build_count = Arc::clone(&build_count);
        let first = thread::spawn(move || {
            with_locked_backend_cache(&first_cache_dir, |_| {
                assert!(!first_ready_marker.exists());
                first_build_count.fetch_add(1, Ordering::SeqCst);
                first_locked_tx.send(()).unwrap();
                release_first_rx.recv().unwrap();
                std::fs::write(first_ready_marker, b"ready").unwrap();
            })
            .unwrap();
        });

        first_locked_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("first cache transaction did not acquire the lock");

        let (second_probe_tx, second_probe_rx) = mpsc::channel();
        let (retry_second_tx, retry_second_rx) = mpsc::channel();
        let (second_done_tx, second_done_rx) = mpsc::channel();
        let second_cache_dir = cache_dir.clone();
        let second_ready_marker = ready_marker.clone();
        let second_build_count = Arc::clone(&build_count);
        let second = thread::spawn(move || {
            let probe = OpenOptions::new()
                .read(true)
                .write(true)
                .open(second_cache_dir.join(BACKEND_CACHE_LOCK_FILE))
                .unwrap();
            let lock_was_held = matches!(probe.try_lock(), Err(std::fs::TryLockError::WouldBlock));
            second_probe_tx.send(lock_was_held).unwrap();
            drop(probe);

            retry_second_rx.recv().unwrap();
            with_locked_backend_cache(&second_cache_dir, |_| {
                let observed_first_result = second_ready_marker.exists();
                if !observed_first_result {
                    second_build_count.fetch_add(1, Ordering::SeqCst);
                    std::fs::write(&second_ready_marker, b"ready").unwrap();
                }
                second_done_tx.send(observed_first_result).unwrap();
            })
            .unwrap();
        });

        let second_observed_os_lock = second_probe_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("second cache transaction did not probe the lock");
        release_first_tx.send(()).unwrap();
        first.join().unwrap();
        retry_second_tx.send(()).unwrap();
        second.join().unwrap();
        let second_observed_first_result = second_done_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("second cache transaction did not finish");

        assert!(
            second_observed_os_lock,
            "a concurrent first-run process must observe the OS lock"
        );
        assert!(
            second_observed_first_result,
            "the second holder must double-check and reuse the first result"
        );
        assert_eq!(
            build_count.load(Ordering::SeqCst),
            1,
            "concurrent first-run transactions must publish exactly one build"
        );

        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[test]
    fn source_fetch_checks_out_exact_revision_not_repository_head() {
        let root = tempdir();
        let repository = root.join("repository");
        std::fs::create_dir_all(&repository).unwrap();
        test_git(&repository, &["init", "--quiet"]);
        test_git(&repository, &["config", "user.name", "cuda-oxide tests"]);
        test_git(
            &repository,
            &["config", "user.email", "cuda-oxide-tests@example.invalid"],
        );

        std::fs::write(repository.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
        std::fs::write(repository.join("revision.txt"), "first\n").unwrap();
        test_git(&repository, &["add", "Cargo.toml", "revision.txt"]);
        test_git(&repository, &["commit", "--quiet", "-m", "first"]);
        let pinned_revision = test_git(&repository, &["rev-parse", "HEAD"]);

        std::fs::write(repository.join("revision.txt"), "second\n").unwrap();
        test_git(&repository, &["add", "revision.txt"]);
        test_git(&repository, &["commit", "--quiet", "-m", "second"]);
        let moving_head = test_git(&repository, &["rev-parse", "HEAD"]);
        assert_ne!(pinned_revision, moving_head);

        let checkout = root.join("checkout");
        fetch_source_at_revision(repository.to_str().unwrap(), &pinned_revision, &checkout)
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(checkout.join("revision.txt"))
                .unwrap()
                .replace("\r\n", "\n"),
            "first\n"
        );
        assert!(source_checkout_matches_revision_at(
            &checkout,
            &pinned_revision
        ));

        std::fs::write(checkout.join("revision.txt"), "modified\n").unwrap();
        assert!(
            !source_checkout_matches_revision_at(&checkout, &pinned_revision),
            "a dirty cache checkout must never qualify as exact source"
        );
    }

    #[test]
    fn windows_backend_build_command_uses_release_profile() {
        let root = tempdir();
        let codegen = root.join("codegen");
        let command = backend_build_command(&codegen, None, "x86_64-pc-windows-msvc");
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert!(args.iter().any(|arg| arg == "--release"));
    }

    #[test]
    fn windows_executable_lookup_accepts_exe_suffix_from_path() {
        let root = tempdir();
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("lld-link.exe"), "").unwrap();
        let paths = std::env::join_paths([bin]).unwrap();

        assert!(windows_executable_in_path("lld-link", &paths));
        assert!(windows_executable_in_path("lld-link.exe", &paths));
        assert!(!windows_executable_in_path("link", &paths));
    }

    #[test]
    fn backend_artifact_uses_cargos_host_path_not_a_stale_legacy_path() {
        let root = tempdir();
        let codegen = root.join("crates/rustc-codegen-cuda");
        std::fs::create_dir_all(&codegen).unwrap();
        std::fs::write(
            codegen.join("Cargo.toml"),
            "[package]\nname='rustc_codegen_cuda'\n",
        )
        .unwrap();

        let stale = codegen.join("target/debug").join(backend_dylib_filename());
        let fresh = codegen
            .join("target/x86_64-unknown-linux-gnu/debug")
            .join(backend_dylib_filename());
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::create_dir_all(fresh.parent().unwrap()).unwrap();
        std::fs::write(&stale, b"stale backend").unwrap();
        std::fs::write(&fresh, b"fresh host backend").unwrap();

        let message = serde_json::json!({
            "reason": "compiler-artifact",
            "manifest_path": codegen.join("Cargo.toml"),
            "target": {
                "kind": ["dylib"],
                "name": "rustc_codegen_cuda"
            },
            "filenames": [fresh]
        });
        let output = format!("{message}\n");

        assert_eq!(
            backend_artifact_from_cargo_output(&codegen, output.as_bytes()).unwrap(),
            fresh
        );

        let no_artifact = b"{\"reason\":\"build-finished\",\"success\":true}\n";
        assert!(
            backend_artifact_from_cargo_output(&codegen, no_artifact).is_err(),
            "a stale target/debug dylib must never be used when Cargo did not report it"
        );
    }

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
    /// the binary signal wins so recovery re-fetches pinned source rather than
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

    /// With no cached backend, `doctor` must say so rather than warn: an
    /// external project simply fetches and builds on first use.
    #[test]
    fn cache_report_is_absent_when_nothing_is_cached() {
        let dir = tempdir();
        let local = dir.join("local.so");
        std::fs::write(&local, b"built").unwrap();

        assert_eq!(
            compare_cache_to_local(&dir.join("missing.so"), &local),
            CacheReport::Absent
        );
    }

    /// A cache older than the local build is the case this reporting exists
    /// for: in-repo commands use the local build and external projects load
    /// the older cached one, with nothing else flagging the difference.
    #[test]
    fn cache_report_is_older_when_the_local_build_is_newer() {
        let dir = tempdir();
        let base = SystemTime::now() - Duration::from_secs(365 * 24 * 60 * 60);
        let cached = dir.join("cached.so");
        let local = dir.join("local.so");
        write_with_mtime(&cached, b"old", base);
        write_with_mtime(&local, b"new", base + Duration::from_secs(60));

        assert_eq!(
            compare_cache_to_local(&cached, &local),
            CacheReport::OlderThanLocal
        );
    }

    /// A cache at least as new as the local build needs no warning.
    #[test]
    fn cache_report_is_up_to_date_when_the_cache_is_newer() {
        let dir = tempdir();
        let base = SystemTime::now() - Duration::from_secs(365 * 24 * 60 * 60);
        let cached = dir.join("cached.so");
        let local = dir.join("local.so");
        write_with_mtime(&local, b"old", base);
        write_with_mtime(&cached, b"new", base + Duration::from_secs(60));

        assert_eq!(
            compare_cache_to_local(&cached, &local),
            CacheReport::UpToDate
        );
    }

    /// Nothing built locally means there is no newer backend to compare
    /// against, so the cache is not stale relative to this checkout.
    #[test]
    fn cache_report_is_up_to_date_when_nothing_is_built_locally() {
        let dir = tempdir();
        let cached = dir.join("cached.so");
        std::fs::write(&cached, b"cached").unwrap();

        assert_eq!(
            compare_cache_to_local(&cached, &dir.join("absent.so")),
            CacheReport::UpToDate
        );
    }

    /// Installing must leave both the `.so` and the toolchain fingerprint in
    /// the cache. A `.so` written without a fingerprint defers to the mtime
    /// checks, which cannot see a toolchain swap, so the next lookup would
    /// load a backend linked against a `librustc_driver` that no longer
    /// resolves.
    #[test]
    fn installing_writes_both_the_backend_and_its_fingerprint() {
        let dir = tempdir();
        let source = dir.join("built.so");
        std::fs::write(&source, b"built").unwrap();

        let cache = dir.join("cache");
        let backend_filename = "backend.bin";
        let installed =
            install_backend_into(&cache, backend_filename, &source).expect("install must succeed");

        assert_eq!(
            installed,
            cache.join(backend_filename),
            "the backend must land under the cache directory"
        );
        assert_eq!(
            std::fs::read(&installed).unwrap(),
            b"built",
            "the installed backend must be the one that was built"
        );

        // Only assert the fingerprint when a rustc is present to produce one;
        // `write_toolchain_fingerprint` is best effort by design.
        if current_toolchain_fingerprint().is_some() {
            assert!(
                cache.join(TOOLCHAIN_FINGERPRINT_FILE).exists(),
                "installing must record the toolchain fingerprint"
            );
        }
    }

    /// Installing into a cache that already holds an older backend must
    /// replace it. This is the case `cargo oxide setup` hits on every run
    /// after the first.
    #[test]
    fn installing_replaces_an_existing_cached_backend() {
        let dir = tempdir();
        let cache = dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let backend_filename = "backend.bin";
        std::fs::write(cache.join(backend_filename), b"stale").unwrap();

        let source = dir.join("built.so");
        std::fs::write(&source, b"fresh").unwrap();

        let installed =
            install_backend_into(&cache, backend_filename, &source).expect("install must succeed");

        assert_eq!(
            std::fs::read(&installed).unwrap(),
            b"fresh",
            "an existing cached backend must be overwritten, not kept"
        );
    }

    /// Publishing a backend that already occupies the shared cache path is a
    /// successful no-op. This is the standalone setup path after discovery
    /// has resolved the cached backend itself.
    #[test]
    fn installing_an_already_cached_backend_is_idempotent() {
        let dir = tempdir();
        let cache = dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        let backend_filename = "backend.bin";
        let source = cache.join(backend_filename);
        std::fs::write(&source, b"built").unwrap();

        let installed =
            install_backend_into(&cache, backend_filename, &source).expect("install must succeed");

        assert_eq!(installed, source);
        assert_eq!(std::fs::read(&installed).unwrap(), b"built");
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
    /// `.so` must be rebuilt from re-fetched pinned source regardless of why
    /// else it is stale.
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

    fn test_git(repository: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(repository)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
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
