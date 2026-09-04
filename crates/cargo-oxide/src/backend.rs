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
//! 3. Backend dynamic library next to the running cargo-oxide executable
//! 4. Local repository backend build
//! 5. Cached backend matching the project's resolved cuda-oxide dependency
//! 6. Build from that dependency's checkout, or the pinned fork revision
//!    when the project has no cuda-oxide dependency.
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
//! we record the fingerprint (`rustc -vV`) of the toolchain that built the
//! `.so` (resolved from the backend source directory, exactly like the build
//! command itself) next to the cached `.so`, and compare it on every lookup
//! against the toolchain active in the user's cwd; a recorded fingerprint
//! that differs from the active toolchain forces a fresh re-clone and rebuild.
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
//!
//! ## Non-convergent toolchain mismatches
//!
//! Re-cloning can only heal a mismatch when upstream's pin agrees with the
//! user's active compiler.
//! When the user's project and the backend source genuinely pin DIFFERENT
//! toolchains, every rebuild re-records the same mismatching fingerprint and a
//! naive retry loops on a multi-minute cold rebuild per invocation. To stop
//! that, each heal attempt first records the (active, recorded) fingerprint
//! pair in a marker file next to the cached `.so`; if the very same pair
//! mismatches again after a rebuild, the lookup reports both toolchain
//! identities with guidance and exits instead of rebuilding. Any lookup that
//! passes the fingerprint check deletes the marker, so a genuinely healed
//! cache clears the memory.
//!
//! ## The backend follows the dependency
//!
//! Outside the repository, the backend source is the checkout Cargo made for
//! the project's `cuda-device` / `cuda-host` dependency (see
//! [`crate::backend_source`]), so the crates a kernel compiles against and
//! the backend that lowers it always come from one commit. That commit is
//! recorded next to the cached `.so` (`source-rev.txt`) and compared on every
//! lookup; a project resolving a different commit rebuilds the cache from its
//! own checkout. A path dependency is a local checkout and builds in place,
//! the same way step 3 does. Dependency checkouts build into
//! `~/.cargo/cuda-oxide/target`, one tree cargo-oxide owns (delete it to
//! reclaim the space) that consecutive commits share their dependency builds
//! in, rather than into Cargo's checkout.
//!
//! Under `cargo oxide` the rustup proxy exports `RUSTUP_TOOLCHAIN` for the
//! project's toolchain, and every child `cargo`/`rustc`, the backend build
//! included, uses it; a checkout's own `rust-toolchain.toml` gets no say.
//! That file still states the nightly the commit was written for, and
//! rustc_private APIs change between nightlies, so before any build the
//! project's active toolchain is compared with that channel and a mismatch is
//! reported up front, naming the channel to set, instead of spending minutes
//! on a backend that fails to compile or to load. The toolchain heal marker
//! above guards only the `main` clone path: a rebuild from a pinned
//! dependency's checkout converges by construction.

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
pub(crate) const PINNED_SOURCE_REVISION: &str = "f5d11395c27069120929e5a8e29c9d4e61feeeed";

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

use crate::backend_source::{self, CODEGEN_CRATE_SUBDIR, DependencySource};

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
/// 3. Packaged backend next to the running executable
/// 4. Local repository backend build
/// 5. Cache matching the project's resolved dependency
/// 6. Build from that dependency, or fetch the pinned fork revision
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

    // Standalone projects build the backend from their resolved dependency.
    standalone_backend(workspace_root)
}

/// Backend for a project outside the repository (discovery steps 4 and 5).
///
/// The project's `Cargo.lock` decides the commit: the backend is built from
/// the checkout Cargo made for the `cuda-device` / `cuda-host` dependency, and
/// the shared cache is reused only when it records that same commit. See the
/// module-level comment ("The backend follows the dependency").
fn standalone_backend(project_dir: &Path) -> PathBuf {
    let source = match backend_source::resolve_dependency_source(project_dir, false) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("Error: could not resolve this project's cuda-oxide dependency: {error}");
            eprintln!(
                "Every cuda-oxide crate in Cargo.toml must come from one git commit or one \
                 local checkout. To bypass this, point CUDA_OXIDE_BACKEND at a backend built \
                 for this project."
            );
            std::process::exit(1);
        }
    };

    if let Some(source) = &source {
        refuse_unloadable_backend(source.checkout(), &source.describe());
        if source.rev().is_none() {
            return build_backend_from_source(&source.codegen_crate());
        }
    }

    let cache_dir = cache_directory().unwrap_or_else(|| {
        eprintln!("Error: Cannot determine cache directory.");
        eprintln!("Set CARGO_HOME or HOME environment variable.");
        std::process::exit(1);
    });
    let backend_filename = backend_filename_for_target(&active_host_target());
    with_locked_backend_cache(&cache_dir, |cache_dir| {
        let source_dir = cache_dir.join("src").join(CODEGEN_CRATE_SUBDIR);
        let expected_rev = source.as_ref().and_then(DependencySource::rev);
        if source.is_none() && !source_checkout_matches_revision(&cache_dir.join("src")) {
            invalidate_cache(cache_dir, &backend_filename);
        }
        if let Some(cached) = consult_backend_cache(
            cache_dir,
            source.is_none().then_some(source_dir.as_path()),
            expected_rev,
        ) {
            return cached;
        }
        match &source {
            Some(source) => build_and_cache(source, cache_dir),
            None => auto_fetch_and_build(cache_dir, &backend_filename),
        }
    })
    .unwrap_or_else(|error| {
        eprintln!("Failed to lock the cuda-oxide backend cache: {error}");
        std::process::exit(1);
    })
}

/// Builds the backend from the dependency's checkout and installs it into the
/// shared cache with its commit recorded. The caller has already run
/// [`refuse_unloadable_backend`] for this checkout.
fn build_and_cache(source: &DependencySource, cache_dir: &Path) -> PathBuf {
    let backend_filename = backend_filename_for_target(&active_host_target());
    eprintln!(
        "Building the cuda-oxide backend from {}...",
        source.describe()
    );
    let codegen_crate = source.codegen_crate();
    // The build tree goes beside the cache, not into Cargo's checkout: one
    // place cargo-oxide owns, shared by every commit's dependency builds.
    let built_so = build_backend_from_source_in(&codegen_crate, Some(&cache_dir.join("target")));
    let so_path = install_backend_into(
        cache_dir,
        &backend_filename,
        &built_so,
        &codegen_crate,
        source.rev(),
    )
    .unwrap_or_else(|error| {
        eprintln!(
            "Error: could not copy the backend into {}: {error}",
            cache_dir.display()
        );
        std::process::exit(1);
    });
    eprintln!("✓ Backend cached at {}", so_path.display());
    so_path
}

/// Refuses to build a backend the project could not build or load.
///
/// Under `cargo oxide` the rustup proxy exports `RUSTUP_TOOLCHAIN` for the
/// project's toolchain, so every child `cargo`/`rustc`, the backend build
/// included, uses that toolchain and the checkout's nested
/// `rust-toolchain.toml` gets no say (it would with a bare `cargo-oxide`
/// binary, which is not how users run it). That file still names the nightly
/// the commit was written for, and rustc_private APIs change between
/// nightlies: any other toolchain fails to compile the backend or builds one
/// its rustc cannot load. So compare the active toolchain with that channel
/// first and say which channel to set. Conservative: without rustup or
/// without a toolchain file in the checkout, let the build proceed and surface
/// its own errors.
fn refuse_unloadable_backend(checkout: &Path, description: &str) {
    let Some(report) = unloadable_backend_report(
        description,
        active_toolchain().as_deref(),
        backend_source::pinned_channel(checkout).as_deref(),
    ) else {
        return;
    };
    eprint!("{report}");
    std::process::exit(1);
}

/// `rustup show active-toolchain` from the process cwd: the toolchain the
/// application build and every child process will use (`RUSTUP_TOOLCHAIN`
/// from the proxy, else the project's `rust-toolchain.toml`, else rustup's
/// default). `None` without rustup.
fn active_toolchain() -> Option<String> {
    let output = Command::new("rustup")
        .args(["show", "active-toolchain"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|active| !active.is_empty())
}

/// The refusal text for a project whose active toolchain does not match the
/// channel the checkout needs, or `None` when it does or either side is
/// unknown. Pure, so the decision and the wording are testable without a
/// process exit.
fn unloadable_backend_report(
    description: &str,
    active: Option<&str>,
    needed_channel: Option<&str>,
) -> Option<String> {
    let (Some(active), Some(channel)) = (active, needed_channel) else {
        return None;
    };
    if crate::commands::active_toolchain_matches_channel(active, channel) {
        return None;
    }
    let active_name = active
        .lines()
        .next()
        .unwrap_or(active)
        .split_whitespace()
        .next()
        .unwrap_or(active);
    let lines = [
        format!(
            "Error: {description} needs Rust `{channel}` (its rust-toolchain.toml), but this \
             project is using `{active_name}`."
        ),
        "The backend is a rustc plugin: it is built with, and only loads into, the toolchain \
         this project uses, and rustc_private APIs change between nightlies."
            .to_string(),
        format!(
            "Set `channel = \"{channel}\"` in this project's rust-toolchain.toml to match \
             {description}, then re-run. To use a backend built some other way, point \
             CUDA_OXIDE_BACKEND at it."
        ),
    ];
    Some(format!("{}\n", lines.join("\n")))
}

/// Consults the shared backend cache at `cache_dir` (discovery step 4).
///
/// `source_dir` is the backend source the cache was built from when that
/// source can change in place (the `main` clone); `expected_rev` is the commit
/// the project's dependency resolves to, when it has one. Returns the cached
/// `.so` when it is fresh. Returns `None` when the caller should fall through
/// to a rebuild, after performing whichever invalidation the staleness verdict
/// calls for. Exits the process with guidance when a toolchain mismatch
/// already failed one heal attempt and rebuilding again cannot converge (see
/// [`toolchain_heal_decision`]).
fn consult_backend_cache(
    cache_dir: &Path,
    source_dir: Option<&Path>,
    expected_rev: Option<&str>,
) -> Option<PathBuf> {
    let backend_filename = backend_filename_for_target(&active_host_target());
    let cached_so = cache_dir.join(&backend_filename);
    if !cached_so.exists() {
        return None;
    }
    match cached_backend_status(&cached_so, source_dir, expected_rev) {
        CacheStatus::Fresh => {
            // The fingerprint check passed: end any heal cycle, so a future,
            // unrelated mismatch gets its own one-shot heal attempt.
            clear_heal_marker(cache_dir);
            Some(cached_so)
        }
        CacheStatus::StaleVsBinary => {
            invalidate_cache(cache_dir, &backend_filename);
            None
        }
        CacheStatus::StaleVsToolchain if expected_rev.is_some() => {
            // The rebuild source is the project's own checkout, and the caller
            // already confirmed the project's toolchain is the one that commit
            // needs, so this rebuild converges. The heal marker guards only
            // the `main` clone path; drop any it left behind. The old `.so`
            // stays until the replacement is installed over it.
            eprintln!(
                "Cached backend was built against a different Rust toolchain; rebuilding \
                 from the project's dependency."
            );
            clear_heal_marker(cache_dir);
            None
        }
        CacheStatus::StaleVsToolchain => match toolchain_heal_decision(cache_dir) {
            ToolchainHealDecision::Heal => {
                eprintln!(
                    "Cached backend was built against a different Rust \
                     toolchain; re-cloning and rebuilding at {}.",
                    cache_dir.display()
                );
                invalidate_cache(cache_dir, &backend_filename);
                None
            }
            ToolchainHealDecision::GiveUp { current, recorded } => {
                report_unhealable_toolchain_mismatch(&current, &recorded);
                std::process::exit(1);
            }
        },
        CacheStatus::StaleVsDependency => {
            // Fall through to a rebuild without deleting anything: the old
            // `.so` and its record stay consistent for whichever project they
            // belong to until `install_backend_into` replaces both, so a
            // failed build never leaves the cache empty.
            let recorded = recorded_source_rev(cache_dir)
                .map(|rev| format!("cuda-oxide {}", backend_source::short_rev(&rev)))
                .unwrap_or_else(|| "an unrecorded cuda-oxide commit".to_string());
            eprintln!(
                "Cached backend was built from {recorded}, but this project depends on \
                 cuda-oxide {}; rebuilding from the project's dependency.",
                backend_source::short_rev(expected_rev.unwrap_or("?"))
            );
            None
        }
        CacheStatus::StaleVsSource => {
            // The cached source advanced; rebuild the `.so` from it in
            // place. We do NOT invalidate the cache here, so the
            // auto-fetch step below skips the clone (the source tree is
            // still present) and rebuilds from the existing source.
            eprintln!(
                "Cached backend source at {} is newer than the cached \
                 library; rebuilding from it in place.",
                source_dir
                    .map(|dir| dir.display().to_string())
                    .unwrap_or_default()
            );
            None
        }
    }
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
/// 3. Packaged backend next to the running cargo-oxide executable.
/// 4. Local repository or path dependency's host build path.
/// 5. Cache path at ~/.cargo/cuda-oxide/<platform filename>.
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

    // A path dependency builds in place (see `standalone_backend`), so its
    // artifact is that checkout's own host build, not the shared cache.
    // Read-only resolution: never fetches or writes; any failure (no
    // Cargo.lock yet, no cargo) falls through to the cache path.
    if let Ok(Some(source)) = backend_source::resolve_dependency_source(workspace_root, true)
        && source.rev().is_none()
    {
        return backend_so_path_candidate(&source.codegen_crate());
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
    /// The cached `.so` was built from a different cuda-oxide commit than the
    /// one the project's dependency resolves to (or from an unrecorded one).
    /// It loads fine but lowers the project's kernels with the wrong backend,
    /// so it must be rebuilt from the project's own checkout.
    StaleVsDependency,
}

/// Classifies the cached backend `.so` against the active toolchain, the
/// commit the project's dependency resolves to (`expected_rev`), the running
/// `cargo-oxide` binary (the user upgraded the binary) and the newest backend
/// source input (the developer advanced the source). When `source_dir` is
/// `None`, or no source inputs can be found under it, the source check is
/// skipped; when `expected_rev` is `None`, the commit check is skipped.
/// Precedence when several fire: toolchain, dependency, binary, source. The
/// first two make the cache wrong for this project outright; the binary check
/// wants a fresh build regardless of source mtimes.
///
/// Conservative on errors: if we can't stat the cached `.so`, we report
/// [`CacheStatus::Fresh`] so a working cache is never invalidated on a failed
/// metadata read.
fn cached_backend_status(
    cached_so: &Path,
    source_dir: Option<&Path>,
    expected_rev: Option<&str>,
) -> CacheStatus {
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

    // Dependency check: the cache must come from the commit this project
    // compiles its kernels against.
    if let (Some(cache_dir), Some(expected)) = (cached_so.parent(), expected_rev)
        && dependency_rev_mismatch(cache_dir, expected)
    {
        return CacheStatus::StaleVsDependency;
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
///
/// "Active" means resolved from the process working directory, i.e. the
/// toolchain the APPLICATION build (which loads the `.so`) will use. The
/// fingerprint RECORDED next to the `.so` must instead come from
/// [`toolchain_fingerprint_in`] with the backend build directory: rustup's
/// `rustc` proxy resolves `rust-toolchain.toml` by walking up from the cwd,
/// and the backend builds with `current_dir = <source clone>` whose nested
/// pin can differ from the user's cwd.
fn current_toolchain_fingerprint() -> Option<String> {
    fingerprint_from_command(toolchain_fingerprint_command(None))
}

/// The fingerprint of the toolchain rustup resolves FROM `build_dir`: the
/// toolchain `backend_build_command` actually builds the `.so` with (same
/// cwd, inherited env, so `RUSTUP_TOOLCHAIN` and the directory's
/// `rust-toolchain.toml` resolve identically).
fn toolchain_fingerprint_in(build_dir: &Path) -> Option<String> {
    fingerprint_from_command(toolchain_fingerprint_command(Some(build_dir)))
}

/// `rustc -vV`, optionally resolved from `build_dir` instead of the process
/// working directory. Split out so tests can assert the resolution cwd.
fn toolchain_fingerprint_command(build_dir: Option<&Path>) -> Command {
    let mut cmd = Command::new("rustc");
    cmd.args(["-vV"]);
    if let Some(dir) = build_dir {
        cmd.current_dir(dir);
    }
    cmd
}

fn fingerprint_from_command(mut cmd: Command) -> Option<String> {
    let output = cmd.output().ok()?;
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

/// Records the fingerprint of the toolchain THAT BUILT the backend (resolved
/// from `build_dir`, the backend source crate) next to the cached `.so`. Best
/// effort: a write failure just means the next run re-detects a mismatch and
/// rebuilds again.
///
/// Recording the user's-cwd toolchain here instead would be a bug: when the
/// user's project pins a different nightly than the source clone, the cached
/// `.so` (linked against the clone's `librustc_driver`) would carry the
/// project's fingerprint, `toolchain_fingerprint_mismatch` would compare the
/// project toolchain against itself and never fire, and every application
/// build would loop on "couldn't load codegen backend" with no self-heal.
fn write_toolchain_fingerprint(cache_dir: &Path, build_dir: &Path) {
    if let Some(fp) = toolchain_fingerprint_in(build_dir) {
        let _ = std::fs::write(cache_dir.join(TOOLCHAIN_FINGERPRINT_FILE), fp);
    }
}

/// File next to the cached `.so` recording the cuda-oxide commit it was built
/// from (the full hash).
const SOURCE_REV_FILE: &str = "source-rev.txt";

/// The commit recorded next to the cached `.so`, if any.
fn recorded_source_rev(cache_dir: &Path) -> Option<String> {
    std::fs::read_to_string(cache_dir.join(SOURCE_REV_FILE))
        .ok()
        .map(|rev| rev.trim().to_string())
        .filter(|rev| !rev.is_empty())
}

/// The commit the shared cache's backend was built from, for `doctor`.
pub fn cached_backend_source_rev() -> Option<String> {
    cache_directory().and_then(|dir| recorded_source_rev(&dir))
}

/// True when the cache was not built from `expected`. An unrecorded commit
/// counts as a mismatch: such a cache predates this check (it was cloned from
/// `main` at an unknown commit), and the one rebuild this triggers records
/// the commit, so it cannot thrash.
fn dependency_rev_mismatch(cache_dir: &Path, expected: &str) -> bool {
    recorded_source_rev(cache_dir).as_deref() != Some(expected)
}

/// Records the commit the installed `.so` was built from, or forgets a stale
/// record when the commit is unknown (a `main` clone, a checkout without git),
/// so a later project can never match against a commit the cache did not come
/// from. Best effort, like the fingerprint: a lost record costs one rebuild.
fn record_source_rev(cache_dir: &Path, source_rev: Option<&str>) {
    let path = cache_dir.join(SOURCE_REV_FILE);
    match source_rev {
        Some(rev) => {
            let _ = std::fs::write(path, rev);
        }
        None => {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// `HEAD` of the git repository containing `dir`, when there is one.
///
/// Used when the in-repo build is published to the shared cache. A dirty tree
/// is recorded under its HEAD on purpose: the developer who runs `cargo oxide
/// setup` wants projects on that commit to pick up the build they just made.
fn repository_head(dir: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(dir)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|rev| !rev.is_empty())
}

/// File next to the cached `.so` recording the (active, recorded) fingerprint
/// pair that triggered the most recent toolchain heal attempt (re-clone +
/// rebuild). Survives [`invalidate_cache`] on purpose: it is the memory that
/// tells the NEXT lookup whether that heal converged.
const TOOLCHAIN_HEAL_MARKER_FILE: &str = "toolchain-heal-attempt.txt";

/// What the `StaleVsToolchain` arm of the cache lookup should do.
#[derive(Debug, PartialEq, Eq)]
enum ToolchainHealDecision {
    /// First mismatch for this (active, recorded) pair: invalidate and
    /// rebuild. This is the legitimate self-heal case, e.g. upstream main
    /// moved to the nightly the user's project just pinned.
    Heal,
    /// A previous heal attempt already re-cloned and rebuilt for this exact
    /// pair and the mismatch persisted: the user's project pin and the
    /// backend source's nested pin genuinely differ, so rebuilding again
    /// would cold-rebuild for minutes on every invocation, forever.
    GiveUp { current: String, recorded: String },
}

/// Decides whether a `StaleVsToolchain` cache may attempt another heal, and
/// records the mismatch pair before approving one so the next IDENTICAL
/// mismatch is recognized as non-convergent. Conservative: when either
/// fingerprint cannot be read, always heal (the pre-guard behavior); the
/// marker write is best effort, a failure just means one more heal attempt.
fn toolchain_heal_decision(cache_dir: &Path) -> ToolchainHealDecision {
    let (Some(current), Ok(recorded)) = (
        current_toolchain_fingerprint(),
        std::fs::read_to_string(cache_dir.join(TOOLCHAIN_FINGERPRINT_FILE)),
    ) else {
        return ToolchainHealDecision::Heal;
    };
    let recorded = recorded.trim().to_string();
    let marker = heal_marker_content(&current, &recorded);
    if std::fs::read_to_string(cache_dir.join(TOOLCHAIN_HEAL_MARKER_FILE))
        .is_ok_and(|stored| stored == marker)
    {
        return ToolchainHealDecision::GiveUp { current, recorded };
    }
    let _ = std::fs::write(cache_dir.join(TOOLCHAIN_HEAL_MARKER_FILE), marker);
    ToolchainHealDecision::Heal
}

/// Serialized form of a heal-attempt pair. Compared as a whole string, so it
/// needs no parsing on the way back in.
fn heal_marker_content(current: &str, recorded: &str) -> String {
    format!("active toolchain:\n{current}\n\nrecorded toolchain:\n{recorded}\n")
}

/// Forgets any recorded heal attempt. Called whenever the cached fingerprint
/// check passes, so a healed cache does not short-circuit a future mismatch.
fn clear_heal_marker(cache_dir: &Path) {
    let _ = std::fs::remove_file(cache_dir.join(TOOLCHAIN_HEAL_MARKER_FILE));
}

/// One compact identity line (release + commit hash) out of a full
/// `rustc -vV` fingerprint, for the non-convergence report.
fn toolchain_identity_line(fingerprint: &str) -> String {
    let field = |prefix: &str| {
        fingerprint
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .unwrap_or("unknown")
    };
    format!(
        "release {} (commit-hash {})",
        field("release: "),
        field("commit-hash: ")
    )
}

/// The repeated-mismatch report: both toolchain identities plus what to do
/// about it. The caller exits afterwards; rebuilding cannot converge.
fn report_unhealable_toolchain_mismatch(current: &str, recorded: &str) {
    eprintln!(
        "Error: the cached cuda-oxide backend was already re-cloned and rebuilt \
         for this exact toolchain mismatch, and rebuilding it again cannot fix it:"
    );
    eprintln!(
        "  your project resolves:  {}",
        toolchain_identity_line(current)
    );
    eprintln!(
        "  the backend built with: {}",
        toolchain_identity_line(recorded)
    );
    eprintln!(
        "Your project's rust-toolchain.toml pins a different nightly than the \
         cuda-oxide backend source it depends on. Align the pins (and run \
         `cargo oxide update` after changing them), or point CUDA_OXIDE_BACKEND \
         at a backend built with your project's toolchain."
    );
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
    clear_cache_contents(cache_dir, backend_filename);
}

fn clear_cache_contents(cache_dir: &Path, backend_filename: &str) {
    let _ = std::fs::remove_file(cache_dir.join(backend_filename));
    let _ = std::fs::remove_dir_all(cache_dir.join("src"));
    // The record describes the `.so` just removed; without it, `doctor` would
    // still report the commit of a backend that no longer exists.
    let _ = std::fs::remove_file(cache_dir.join(SOURCE_REV_FILE));
}

/// Rebuild the backend for the project in `project_dir` (external projects):
/// from the commit its cuda-oxide dependency resolves to, in place for a path
/// dependency, or from a fresh `main` clone without any dependency. The shared
/// cache is cleared first whenever the project uses it.
///
/// Returns the path to the freshly built `.so`.
pub fn refresh_cached_backend(project_dir: &Path) -> PathBuf {
    // A path dependency builds in place and never touches the shared cache;
    // clearing it would only cost other projects a rebuild. Read-only
    // resolution here; the real one, which may fetch, runs in
    // `standalone_backend`.
    let builds_in_place = matches!(
        backend_source::resolve_dependency_source(project_dir, true),
        Ok(Some(ref source)) if source.rev().is_none()
    );
    if !builds_in_place && let Some(cache_dir) = cache_directory() {
        let backend_filename = backend_filename_for_target(&active_host_target());
        with_locked_backend_cache(&cache_dir, |dir| {
            clear_cache_contents(dir, &backend_filename);
        })
        .unwrap_or_else(|error| {
            eprintln!("Failed to lock backend cache: {error}");
            std::process::exit(1);
        });
    }
    standalone_backend(project_dir)
}

/// Builds the backend from a local source tree, with the build tree at
/// `<codegen_crate>/target`: the layout the in-repo path, `cargo oxide setup`
/// and the passive [`backend_so_path_candidate`] expect.
pub fn build_backend_from_source(codegen_crate: &Path) -> PathBuf {
    build_backend_from_source_in(codegen_crate, None)
}

/// Builds the backend from `codegen_crate`, with the build tree at
/// `target_dir` when given and at `<codegen_crate>/target` otherwise.
/// Dependency checkouts pass `~/.cargo/cuda-oxide/target`: one tree that
/// cargo-oxide owns, that consecutive commits share their dependency builds
/// in (pliron and friends rebuild only when they change), and that a user can
/// delete to reclaim the space, instead of a build tree inside Cargo's
/// checkout that nothing ever cleans up.
pub fn build_backend_from_source_in(codegen_crate: &Path, target_dir: Option<&Path>) -> PathBuf {
    println!("Building rustc-codegen-cuda backend...");
    // The application build still honors these (codegen_env.rs folds them
    // into the composed flags), so say once why the backend build does not.
    let ambient_flags_present = [
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "CARGO_BUILD_RUSTFLAGS",
    ]
    .iter()
    .any(|var| std::env::var(var).is_ok_and(|value| !value.trim().is_empty()));
    if ambient_flags_present {
        println!(
            "  note: ambient RUSTFLAGS are ignored for the backend dylib (its \
             digest keys every build cache); they still apply to application \
             builds. For custom backend flags, build crates/rustc-codegen-cuda \
             manually and point CUDA_OXIDE_BACKEND at the result."
        );
    }

    let host_target = active_host_target();
    let rustc_sysroot = get_rustc_sysroot();
    let loader_path = rustc_sysroot
        .as_ref()
        .map(|s| rustc_sysroot_loader_dir(s, &host_target));

    let mut cmd = backend_build_command_in(
        codegen_crate,
        target_dir,
        loader_path.as_deref(),
        &host_target,
    );
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

fn backend_build_command_in(
    codegen_crate: &Path,
    target_dir: Option<&Path>,
    loader_path: Option<&Path>,
    host_target: &str,
) -> Command {
    let codegen_crate = absolute_path(codegen_crate);
    let target_dir = target_dir
        .map(absolute_path)
        .unwrap_or_else(|| codegen_crate.join("target"));
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

    // Keep application rustflags out of the backend dylib: its digest keys the
    // application cache, so ambient flags would create unstable identities.
    // An empty encoded value also overrides target/config rustflags that
    // cannot be removed directly from this process environment.
    cmd.env("CARGO_ENCODED_RUSTFLAGS", "");
    cmd.env_remove("RUSTFLAGS");
    cmd.env_remove("CARGO_BUILD_RUSTFLAGS");

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
    cache_directory().map(|dir| dir.join(backend_filename_for_target(&active_host_target())))
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

    refuse_unloadable_backend(&src_dir, "the pinned cuda-oxide checkout");
    let codegen_crate = src_dir.join(CODEGEN_CRATE_SUBDIR);
    let built_backend = build_backend_from_source(&codegen_crate);
    if built_backend.exists() {
        install_backend_into(
            cache_dir,
            backend_filename,
            &built_backend,
            &codegen_crate,
            None,
        )
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
/// fingerprint and the source commit beside it.
///
/// The fingerprint must be written whenever the backend is. A backend installed
/// without one falls back to the mtime checks, which cannot see a toolchain
/// swap, so the next lookup would load a backend linked against the wrong
/// `librustc_driver`. The commit record follows the same rule: written with
/// the `.so`, or removed when the commit is unknown (see
/// [`record_source_rev`]).
///
/// Takes the directories explicitly so it can be exercised without touching
/// `CARGO_HOME`. `build_dir` is the backend source crate the `.so` was built
/// in; the recorded fingerprint is resolved from there (see
/// [`write_toolchain_fingerprint`]).
fn install_backend_into(
    cache_dir: &Path,
    backend_filename: &str,
    built_backend: &Path,
    build_dir: &Path,
    source_rev: Option<&str>,
) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(cache_dir)?;
    let backend_path = cache_dir.join(backend_filename);
    let source_is_destination = built_backend == backend_path
        || match (built_backend.canonicalize(), backend_path.canonicalize()) {
            (Ok(source), Ok(destination)) => source == destination,
            _ => false,
        };
    // A failed copy must not retain an obsolete source revision.
    record_source_rev(cache_dir, None);
    if !source_is_destination {
        std::fs::copy(built_backend, &backend_path)?;
    }
    write_toolchain_fingerprint(cache_dir, build_dir);
    record_source_rev(cache_dir, source_rev);
    Ok(backend_path)
}

/// What [`publish_to_cache`] installed.
pub struct PublishedBackend {
    /// The cached `.so`.
    pub path: PathBuf,
    /// The commit recorded beside it, when the checkout is a git repository.
    pub source_rev: Option<String>,
}

/// Publishes a freshly built backend to the shared cache at
/// `~/.cargo/cuda-oxide/`.
///
/// That path is what step 5 of the discovery order resolves to, and it is the
/// only one a project outside this repository can reach: `find_workspace_root`
/// walks up from the current directory looking for `crates/rustc-codegen-cuda`
/// and finds nothing from an unrelated crate. The checkout's HEAD is recorded
/// beside the `.so`, so only projects whose dependency resolves to that commit
/// reuse it.
///
/// Returns `None` when the cache directory cannot be determined or the copy
/// fails. Callers treat this as best effort: a failure leaves the in-repo build
/// usable and costs external projects only a rebuild.
pub fn publish_to_cache(built_so: &Path, codegen_crate: &Path) -> Option<PublishedBackend> {
    let cache_dir = cache_directory()?;
    let backend_filename = backend_filename_for_target(&active_host_target());
    let source_rev = repository_head(codegen_crate);
    let path = with_locked_backend_cache(&cache_dir, |locked_cache_dir| {
        install_backend_into(
            locked_cache_dir,
            &backend_filename,
            built_so,
            codegen_crate,
            source_rev.as_deref(),
        )
    })
    .ok()?
    .ok()?;
    Some(PublishedBackend { path, source_rev })
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
        let command =
            backend_build_command_in(&codegen, None, Some(&rustc_lib), "x86_64-unknown-linux-gnu");

        let cargo_target_dir = command
            .get_envs()
            .find_map(|(key, value)| (key == OsStr::new("CARGO_TARGET_DIR")).then_some(value));
        let cargo_build_target = command
            .get_envs()
            .find_map(|(key, value)| (key == OsStr::new("CARGO_BUILD_TARGET")).then_some(value));
        // Ambient flags must not alter the backend bits, or the identity cfg
        // digest forks every application unit's cache slot. The empty-but-set
        // CARGO_ENCODED_RUSTFLAGS is what silences the sources env_remove
        // cannot reach (CARGO_TARGET_<TRIPLE>_RUSTFLAGS, config-file
        // [build]/[target.*] rustflags).
        let encoded = command.get_envs().find_map(|(key, value)| {
            (key == OsStr::new("CARGO_ENCODED_RUSTFLAGS")).then_some(value)
        });
        assert_eq!(
            encoded,
            Some(Some(OsStr::new(""))),
            "CARGO_ENCODED_RUSTFLAGS must be set to the empty string"
        );
        for scrubbed in ["RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS"] {
            let entry = command
                .get_envs()
                .find_map(|(key, value)| (key == OsStr::new(scrubbed)).then_some(value));
            assert_eq!(entry, Some(None), "{scrubbed} must be scrubbed");
        }
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
        let command = backend_build_command_in(&codegen, None, None, "x86_64-pc-windows-msvc");
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
            cached_backend_status(&so, None, None),
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
            cached_backend_status(&so, None, None),
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
        assert_eq!(cached_backend_status(&so, None, None), CacheStatus::Fresh);
    }

    #[test]
    fn clear_cache_contents_removes_so_src_tree_and_commit_record() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        let src = dir.join("src/crates/rustc-codegen-cuda");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(&so, b"old").unwrap();
        std::fs::write(src.join("lib.rs"), b"fn main() {}").unwrap();
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111").unwrap();

        clear_cache_contents(&dir, "librustc_codegen_cuda.so");

        assert!(!so.exists());
        assert!(!dir.join("src").exists());
        assert!(
            !dir.join(SOURCE_REV_FILE).exists(),
            "a record without its .so would misreport a backend that is gone"
        );
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
            cached_backend_status(&so, Some(&dir), None),
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
            cached_backend_status(&so, Some(&dir), None),
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
            cached_backend_status(&so, Some(&absent), None),
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
            cached_backend_status(&so, Some(&dir), None),
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

    /// The recorded fingerprint must come from the toolchain that BUILT the
    /// backend. `backend_build_command` runs cargo with
    /// `current_dir = <codegen crate>`, so rustup resolves the nested
    /// `rust-toolchain.toml` there; the fingerprint command must resolve from
    /// the same directory. Fingerprinting the user's-cwd rustc instead records
    /// a fingerprint that can never match the `.so`, so the
    /// `StaleVsToolchain` guard never fires and every application build loops
    /// on "couldn't load codegen backend".
    #[test]
    fn fingerprint_command_resolves_from_the_build_dir() {
        let command = toolchain_fingerprint_command(Some(Path::new("/tmp/codegen")));
        assert_eq!(command.get_program(), OsStr::new("rustc"));
        assert_eq!(
            command.get_current_dir(),
            Some(Path::new("/tmp/codegen")),
            "must resolve rustup's rust-toolchain.toml from the backend build dir"
        );
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(args, ["-vV"]);

        // The application-side check keeps resolving from the process cwd.
        let command = toolchain_fingerprint_command(None);
        assert_eq!(command.get_current_dir(), None);
    }

    /// Behavioral proof that resolution honors the build dir: a build dir
    /// pinning a toolchain that cannot resolve must defeat fingerprinting
    /// (and thus `write_toolchain_fingerprint` writes nothing), even though
    /// the process cwd still resolves fine. Skipped when `rustc` is not a
    /// rustup proxy (resolution is then cwd-insensitive by construction).
    #[test]
    fn fingerprint_resolution_follows_the_build_dir_pin_not_the_cwd() {
        if current_toolchain_fingerprint().is_none() {
            return; // no rustc here; nothing to observe
        }
        let dir = tempdir();
        std::fs::write(
            dir.join("rust-toolchain.toml"),
            "[toolchain]\nchannel = \"cuda-oxide-nonexistent-test-toolchain\"\n",
        )
        .unwrap();
        let pinned = toolchain_fingerprint_in(&dir);
        if pinned.is_some() {
            return; // not a rustup proxy; the pin cannot influence resolution
        }

        let cache = dir.join("cache");
        std::fs::create_dir_all(&cache).unwrap();
        write_toolchain_fingerprint(&cache, &dir);
        assert!(
            !cache.join(TOOLCHAIN_FINGERPRINT_FILE).exists(),
            "the recorded fingerprint must be resolved from the build dir, \
             not from the process cwd"
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
        let installed = install_backend_into(&cache, backend_filename, &source, &dir, None)
            .expect("install must succeed");

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

        let installed = install_backend_into(&cache, backend_filename, &source, &dir, None)
            .expect("install must succeed");

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

        let installed = install_backend_into(&cache, backend_filename, &source, &dir, None)
            .expect("install must succeed");

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
            cached_backend_status(&so, None, None),
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
            cached_backend_status(&so, None, None),
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
            cached_backend_status(&so, None, None),
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
            cached_backend_status(&so, None, None),
            CacheStatus::StaleVsToolchain,
            "toolchain mismatch must win over binary staleness"
        );
    }

    /// The heal guard must short-circuit a REPEATED identical mismatch pair:
    /// the first `StaleVsToolchain` verdict for a pair heals (recording the
    /// pair first), the second identical one gives up instead of re-cloning
    /// and cold-rebuilding on every invocation forever. A pair that changed
    /// (a different recorded fingerprint after a rebuild) is a fresh
    /// mismatch and must heal again.
    #[test]
    fn repeated_identical_toolchain_mismatch_gives_up_instead_of_rebuilding() {
        let Some(current) = current_toolchain_fingerprint() else {
            return; // no rustc here; nothing to observe
        };
        let dir = tempdir();
        let first_recorded = "rustc 0.0.0 (deadbeef 1970-01-01)\nrelease: 0.0.0";
        std::fs::write(dir.join(TOOLCHAIN_FINGERPRINT_FILE), first_recorded).unwrap();

        assert_eq!(
            toolchain_heal_decision(&dir),
            ToolchainHealDecision::Heal,
            "the FIRST mismatch for a pair is the legitimate self-heal case"
        );
        assert!(
            dir.join(TOOLCHAIN_HEAL_MARKER_FILE).exists(),
            "a heal attempt must be recorded before the rebuild runs"
        );

        assert_eq!(
            toolchain_heal_decision(&dir),
            ToolchainHealDecision::GiveUp {
                current: current.clone(),
                recorded: first_recorded.to_string(),
            },
            "the SAME pair after a heal attempt cannot converge; it must stop rebuilding"
        );

        // A rebuild that changed the recorded fingerprint (e.g. the source
        // clone advanced to a new pin) is a NEW mismatch: heal once more.
        std::fs::write(
            dir.join(TOOLCHAIN_FINGERPRINT_FILE),
            "rustc 0.0.1 (cafef00d 1970-01-02)",
        )
        .unwrap();
        assert_eq!(
            toolchain_heal_decision(&dir),
            ToolchainHealDecision::Heal,
            "a changed mismatch pair must get its own heal attempt"
        );
    }

    /// A cache lookup that passes the fingerprint check must delete the heal
    /// marker: a genuinely healed cache forgets the old mismatch, so a
    /// future, unrelated mismatch gets its own one-shot heal attempt instead
    /// of being short-circuited by stale memory.
    #[test]
    fn clear_heal_marker_forgets_a_successful_heal() {
        let Some(fp) = current_toolchain_fingerprint() else {
            return; // no rustc here; nothing to assert
        };
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join(backend_filename_for_target(&active_host_target()));
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(TOOLCHAIN_FINGERPRINT_FILE), fp).unwrap();
        std::fs::write(dir.join(TOOLCHAIN_HEAL_MARKER_FILE), "stale heal memory").unwrap();

        assert_eq!(
            consult_backend_cache(&dir, None, None),
            Some(so),
            "a matching fingerprint with fresh mtimes must reuse the cache"
        );
        assert!(
            !dir.join(TOOLCHAIN_HEAL_MARKER_FILE).exists(),
            "a passing fingerprint check must clear the heal marker"
        );
    }

    /// The cache records the cuda-oxide commit it was built from. A project
    /// whose dependency resolves to another commit must not load it, however
    /// fresh the mtimes look: its kernels compile against `cuda-device` at one
    /// commit and would be lowered by a backend from another.
    #[test]
    fn stale_when_cache_was_built_from_another_commit() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111\n").unwrap();

        assert_eq!(
            cached_backend_status(&so, None, Some("bbbb2222")),
            CacheStatus::StaleVsDependency,
            "a cache from another commit must be stale for this project"
        );
    }

    #[test]
    fn fresh_when_cache_matches_the_dependency_commit() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111\n").unwrap();

        assert_eq!(
            cached_backend_status(&so, None, Some("aaaa1111")),
            CacheStatus::Fresh,
            "a cache from the project's own commit must be reused"
        );
    }

    /// A cache with no recorded commit predates this check: it came from a
    /// `main` clone at an unknown commit. When the project pins one, rebuild
    /// once; the rebuild records the commit, so this fires once, not forever.
    #[test]
    fn unrecorded_commit_is_stale_when_the_project_pins_one() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);

        assert_eq!(
            cached_backend_status(&so, None, Some("aaaa1111")),
            CacheStatus::StaleVsDependency,
            "an unrecorded commit cannot be trusted to match the project's"
        );
    }

    /// A project with no cuda-oxide dependency pins nothing, so whatever
    /// commit the cache records is no reason to rebuild.
    #[test]
    fn recorded_commit_is_ignored_when_the_project_pins_none() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111\n").unwrap();

        assert_eq!(
            cached_backend_status(&so, None, None),
            CacheStatus::Fresh,
            "no expected commit means no commit check"
        );
    }

    /// An unloadable `.so` is stale whichever commit it came from, so the
    /// toolchain verdict (and its heal guard) keeps the highest precedence.
    #[test]
    fn toolchain_staleness_takes_precedence_over_dependency() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(
            dir.join(TOOLCHAIN_FINGERPRINT_FILE),
            "rustc 0.0.0 (deadbeef 1970-01-01)",
        )
        .unwrap();
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111\n").unwrap();

        assert_eq!(
            cached_backend_status(&so, None, Some("bbbb2222")),
            CacheStatus::StaleVsToolchain,
            "toolchain mismatch must win over a commit mismatch"
        );
    }

    /// A binary upgrade and a commit change both rebuild from the project's
    /// checkout; the commit verdict wins so the message names the real cause.
    #[test]
    fn dependency_staleness_takes_precedence_over_binary() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Backdate the `.so` so binary-staleness would otherwise fire.
        write_with_mtime(&so, b"built", SystemTime::now() - year);
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111\n").unwrap();

        assert_eq!(
            cached_backend_status(&so, None, Some("bbbb2222")),
            CacheStatus::StaleVsDependency,
            "commit mismatch must win over binary staleness"
        );
    }

    /// Installing writes the commit beside the `.so`, and installing a build
    /// of unknown origin removes any earlier record: a stale record would let
    /// a later project match a cache that did not come from its commit.
    #[test]
    fn installing_records_the_source_commit_and_forgets_an_unknown_one() {
        let dir = tempdir();
        let source = dir.join("built.so");
        std::fs::write(&source, b"built").unwrap();
        let cache = dir.join("cache");

        let filename = backend_filename_for_target(&active_host_target());
        install_backend_into(&cache, &filename, &source, &dir, Some("aaaa1111")).expect("install");
        assert_eq!(
            std::fs::read_to_string(cache.join(SOURCE_REV_FILE)).unwrap(),
            "aaaa1111",
            "installing must record the source commit"
        );

        install_backend_into(&cache, &filename, &source, &dir, None).expect("install");
        assert!(
            !cache.join(SOURCE_REV_FILE).exists(),
            "a build of unknown origin must not keep the previous commit record"
        );
    }

    /// A commit mismatch falls through to a rebuild without deleting anything:
    /// the old `.so` and its record stay consistent for the project they
    /// belong to until the replacement is installed over them, so a failed
    /// build never empties the cache. It must not touch the heal marker
    /// machinery either, which belongs to toolchain mismatches.
    #[test]
    fn dependency_mismatch_keeps_the_old_cache_until_replaced() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join(backend_filename_for_target(&active_host_target()));
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111\n").unwrap();

        assert_eq!(
            consult_backend_cache(&dir, None, Some("bbbb2222")),
            None,
            "a cache from another commit must not be handed out"
        );
        assert!(
            so.exists(),
            "the old backend stays until the replacement lands"
        );
        assert_eq!(
            recorded_source_rev(&dir).as_deref(),
            Some("aaaa1111"),
            "the old record stays with the old backend"
        );
        assert!(
            !dir.join(TOOLCHAIN_HEAL_MARKER_FILE).exists(),
            "a commit mismatch is not a toolchain heal attempt"
        );
    }

    /// A toolchain mismatch with a pinned dependency must not consult the
    /// heal-marker gate: the rebuild comes from the project's own checkout and
    /// converges, so a marker left by the `main` clone path (or by the
    /// previous cargo-oxide, whose users this fix targets) is dropped rather
    /// than turned into a "cannot converge" exit. The old `.so` stays until
    /// the replacement is installed. Without a pinned dependency the gate
    /// still applies.
    #[test]
    fn toolchain_mismatch_with_a_pinned_dependency_rebuilds_instead_of_giving_up() {
        let Some(current) = current_toolchain_fingerprint() else {
            return; // no rustc here; nothing to observe
        };
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join(backend_filename_for_target(&active_host_target()));
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        let recorded = "rustc 0.0.0 (deadbeef 1970-01-01)";
        std::fs::write(dir.join(TOOLCHAIN_FINGERPRINT_FILE), recorded).unwrap();
        std::fs::write(
            dir.join(TOOLCHAIN_HEAL_MARKER_FILE),
            heal_marker_content(&current, recorded),
        )
        .unwrap();

        assert_eq!(
            consult_backend_cache(&dir, None, Some("aaaa1111")),
            None,
            "a toolchain mismatch must fall through to a rebuild from the dependency"
        );
        assert!(
            so.exists(),
            "the old backend stays until the replacement lands"
        );
        assert!(
            !dir.join(TOOLCHAIN_HEAL_MARKER_FILE).exists(),
            "the heal marker belongs to the main-clone path and must be dropped"
        );

        std::fs::write(
            dir.join(TOOLCHAIN_HEAL_MARKER_FILE),
            heal_marker_content(&current, recorded),
        )
        .unwrap();
        assert_eq!(
            toolchain_heal_decision(&dir),
            ToolchainHealDecision::GiveUp {
                current,
                recorded: recorded.to_string(),
            },
            "without a pinned dependency the same marker still means \"already tried\""
        );
    }

    /// Dependency checkouts build into the shared cache's `target`, not into
    /// Cargo's checkout; the in-repo default stays `<crate>/target`.
    #[test]
    fn backend_build_command_honours_a_target_dir_override() {
        let dir = tempdir();
        let codegen = dir.join("codegen");
        let target = dir.join("cache/target");
        let command =
            backend_build_command_in(&codegen, Some(&target), None, &active_host_target());
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            args.windows(2)
                .any(|args| args == ["--target-dir", target.to_str().unwrap()])
        );
        let cargo_target_dir = command
            .get_envs()
            .find_map(|(key, value)| (key == OsStr::new("CARGO_TARGET_DIR")).then_some(value));
        assert_eq!(cargo_target_dir.flatten(), Some(target.as_os_str()));
        assert_eq!(command.get_current_dir(), Some(codegen.as_path()));

        let default = backend_build_command_in(&codegen, None, None, &active_host_target());
        let args = default
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(
            args.windows(2)
                .any(|args| args == ["--target-dir", codegen.join("target").to_str().unwrap()])
        );
    }

    /// The refusal compares the ACTIVE toolchain (what the proxy exported and
    /// what will build and load the backend) with the channel the checkout
    /// needs, fires only when both are known and differ, tolerates rustup's
    /// host-suffixed names, and names the channel to set plus the escape
    /// hatch. Comparing two `rustc -vV` fingerprints instead would never fire
    /// under `cargo oxide`, where `RUSTUP_TOOLCHAIN` makes both resolve alike.
    #[test]
    fn unloadable_backend_report_compares_channels_and_names_the_fix() {
        let dep = "cuda-oxide 596a6353de (git dependency)";
        let april = "nightly-2026-04-03-x86_64-unknown-linux-gnu (overridden by environment \
                     variable RUSTUP_TOOLCHAIN)";
        let august =
            "nightly-2026-08-28-x86_64-unknown-linux-gnu (overridden by '/p/rust-toolchain.toml')";

        assert_eq!(
            unloadable_backend_report(dep, None, Some("nightly-2026-08-28")),
            None,
            "no rustup means no verdict; let the build proceed"
        );
        assert_eq!(
            unloadable_backend_report(dep, Some(april), None),
            None,
            "a checkout without a toolchain file pins nothing to compare with"
        );
        assert_eq!(
            unloadable_backend_report(dep, Some(august), Some("nightly-2026-08-28")),
            None,
            "the host-suffixed active name matches its channel"
        );

        let report = unloadable_backend_report(dep, Some(april), Some("nightly-2026-08-28"))
            .expect("a different channel must be refused");
        assert!(
            report.contains("needs Rust `nightly-2026-08-28`"),
            "{report}"
        );
        assert!(
            report.contains("using `nightly-2026-04-03-x86_64-unknown-linux-gnu`"),
            "the active toolchain must be named without rustup's trailing reason: {report}"
        );
        assert!(
            report.contains(
                "Set `channel = \"nightly-2026-08-28\"` in this project's rust-toolchain.toml"
            ),
            "{report}"
        );
        assert!(report.contains("CUDA_OXIDE_BACKEND"), "{report}");
    }

    /// The commit verdict outranks the source-mtime verdict: the wrong commit
    /// is the wrong backend whatever the mtimes say.
    #[test]
    fn dependency_staleness_takes_precedence_over_source() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(SOURCE_REV_FILE), "aaaa1111\n").unwrap();
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_with_mtime(
            &src.join("lib.rs"),
            b"// newer",
            SystemTime::now() + year + year,
        );

        assert_eq!(
            cached_backend_status(&so, Some(&dir), Some("bbbb2222")),
            CacheStatus::StaleVsDependency,
            "commit mismatch must win over source staleness"
        );
    }

    /// `setup` records the repository HEAD beside the published `.so`, and
    /// `dependency_rev_mismatch` compares it byte for byte with the 40-hex
    /// commit Cargo reports. So it must be the full hash, trimmed, resolved
    /// from a directory inside the repository, and unaffected by a dirty tree.
    #[test]
    fn repository_head_is_the_full_commit_of_the_enclosing_repo() {
        let repo = tempdir();
        let git = |args: &[&str]| {
            Command::new("git")
                .args(["-c", "commit.gpgsign=false"])
                .args(args)
                .current_dir(&repo)
                .output()
        };
        if !git(&["init", "-q"]).is_ok_and(|output| output.status.success()) {
            return; // no git here; nothing to observe
        }
        let committed = git(&[
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ])
        .unwrap();
        assert!(
            committed.status.success(),
            "{}",
            String::from_utf8_lossy(&committed.stderr)
        );
        let expected = String::from_utf8_lossy(&git(&["rev-parse", "HEAD"]).unwrap().stdout)
            .trim()
            .to_string();
        assert_eq!(expected.len(), 40, "git must report the full hash");

        let nested = repo.join(CODEGEN_CRATE_SUBDIR);
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(
            repository_head(&nested),
            Some(expected.clone()),
            "HEAD must resolve from the backend crate directory inside the repo"
        );

        std::fs::write(repo.join("dirty.txt"), b"uncommitted").unwrap();
        assert_eq!(
            repository_head(&nested),
            Some(expected),
            "a dirty tree is recorded under its HEAD"
        );

        let outside = tempdir();
        let in_a_repo = Command::new("git")
            .args(["rev-parse", "--is-inside-work-tree"])
            .current_dir(&outside)
            .output()
            .is_ok_and(|output| output.status.success());
        if !in_a_repo {
            assert_eq!(
                repository_head(&outside),
                None,
                "outside any repository there is nothing to record"
            );
        }
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
