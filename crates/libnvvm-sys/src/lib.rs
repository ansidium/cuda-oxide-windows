/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime (`dlopen`) bindings to NVIDIA's libNVVM.
//!
//! libNVVM is the front-end of NVIDIA's PTX-targeting compiler. It accepts
//! NVVM IR (an LLVM-IR dialect) and produces either PTX or LTOIR.
//!
//! This crate is a thin, RAII Rust binding that loads libNVVM lazily at
//! runtime via `libloading`. It is not a `bindgen`-generated wrapper, so it
//! does not require the CUDA Toolkit to be present at build time, only at run
//! time.
//!
//! # Library discovery
//!
//! [`LibNvvm::load`] tries (in order):
//! 1. `LIBNVVM_PATH` env var, if set.
//! 2. Platform loader names (`libnvvm.so.4`, `libnvvm.so.3`, `libnvvm.so` on
//!    Linux; discovered `nvvm64_*.dll` files on Windows).
//! 3. CUDA Toolkit roots from `cuda-toolkit-discovery`, including
//!    `<root>/nvvm/lib64/libnvvm.so` on Linux and
//!    `<root>/nvvm/bin/x64/nvvm64_*.dll` on Windows.
//!
//! # Symbol naming
//!
//! libNVVM uses plain unversioned symbol names (`nvvmCreateProgram` etc.),
//! so a single `dlsym` / `GetProcAddress` lookup per function is sufficient
//! across CUDA versions.
//!
//! # Example
//!
//! ```no_run
//! use libnvvm_sys::{LibNvvm, Program};
//!
//! let nvvm = LibNvvm::load().expect("CUDA Toolkit (libnvvm) not found");
//! let mut program = Program::new(&nvvm).unwrap();
//! program.add_module(b"; NVVM IR here\n", "kernel").unwrap();
//! let ltoir = program.compile(&["-arch=compute_120", "-gen-lto"]).unwrap();
//! assert!(!ltoir.is_empty());
//! ```

use libloading::{Library, Symbol};
use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr;
use thiserror::Error;

// ============================================================================
// FFI types
// ============================================================================

/// Opaque libNVVM program handle (`nvvmProgram`).
#[repr(transparent)]
#[derive(Copy, Clone)]
struct NvvmProgram(*mut c_void);

type NvvmResultRaw = c_int;

/// Known libNVVM result codes (`nvvmResult`). Mirrors `nvvm.h`.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NvvmResultCode {
    Success,
    OutOfMemory,
    ProgramCreationFailure,
    IrVersionMismatch,
    InvalidInput,
    InvalidProgram,
    InvalidIr,
    InvalidOption,
    NoModuleInProgram,
    CompilationFailure,
}

impl NvvmResultCode {
    fn from_raw(raw: NvvmResultRaw) -> Option<Self> {
        Some(match raw {
            0 => Self::Success,
            1 => Self::OutOfMemory,
            2 => Self::ProgramCreationFailure,
            3 => Self::IrVersionMismatch,
            4 => Self::InvalidInput,
            5 => Self::InvalidProgram,
            6 => Self::InvalidIr,
            7 => Self::InvalidOption,
            8 => Self::NoModuleInProgram,
            9 => Self::CompilationFailure,
            _ => return None,
        })
    }

    fn as_raw(self) -> NvvmResultRaw {
        match self {
            Self::Success => 0,
            Self::OutOfMemory => 1,
            Self::ProgramCreationFailure => 2,
            Self::IrVersionMismatch => 3,
            Self::InvalidInput => 4,
            Self::InvalidProgram => 5,
            Self::InvalidIr => 6,
            Self::InvalidOption => 7,
            Self::NoModuleInProgram => 8,
            Self::CompilationFailure => 9,
        }
    }
}

impl fmt::Display for NvvmResultCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => f.write_str("Success"),
            Self::OutOfMemory => f.write_str("OutOfMemory"),
            Self::ProgramCreationFailure => f.write_str("ProgramCreationFailure"),
            Self::IrVersionMismatch => f.write_str("IrVersionMismatch"),
            Self::InvalidInput => f.write_str("InvalidInput"),
            Self::InvalidProgram => f.write_str("InvalidProgram"),
            Self::InvalidIr => f.write_str("InvalidIr"),
            Self::InvalidOption => f.write_str("InvalidOption"),
            Self::NoModuleInProgram => f.write_str("NoModuleInProgram"),
            Self::CompilationFailure => f.write_str("CompilationFailure"),
        }
    }
}

// ============================================================================
// Errors
// ============================================================================

/// All errors surfaced by this crate.
#[derive(Debug, Error)]
pub enum NvvmError {
    /// libNVVM could not be located on this system. `tried` lists every path,
    /// loader name, or search pattern that was probed, in order, joined by
    /// newlines.
    #[error(
        "libNVVM could not be located. Set LIBNVVM_PATH or a CUDA Toolkit root, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibraryNotFound {
        /// Newline-joined list of paths, loader names, and search patterns.
        tried: String,
    },

    /// libNVVM was loaded, but `dlsym` failed to resolve a function this
    /// crate requires. Indicates an old or broken libNVVM that does not
    /// export the standard NVVM IR API.
    #[error("libNVVM was found but a required symbol is missing: {symbol}: {source}")]
    SymbolNotFound {
        /// Name of the missing libNVVM function (e.g. `nvvmCreateProgram`).
        symbol: &'static str,
        /// Underlying `libloading` error returned by `dlsym`.
        #[source]
        source: libloading::Error,
    },

    /// A libNVVM call returned a known non-`Success` `nvvmResult`. `log`
    /// carries the libNVVM program log when it is available, or the
    /// `nvvmGetErrorString` text otherwise.
    #[error("libNVVM error in {operation}: {code}{}", .log.as_ref().map(|l| format!("\n--- libNVVM log ---\n{l}")).unwrap_or_default())]
    Call {
        /// Name of the libNVVM function that failed.
        operation: &'static str,
        /// Known `nvvmResult` value.
        code: NvvmResultCode,
        /// Best-effort error message: program log first, then
        /// `nvvmGetErrorString`. `None` only if both were unavailable.
        log: Option<String>,
    },

    /// A libNVVM call returned an integer that does not map to any known
    /// `nvvmResult` value. The raw value is preserved without constructing an
    /// invalid Rust enum.
    #[error("libNVVM returned unknown nvvmResult in {operation}: {code}")]
    UnknownResult {
        /// Name of the libNVVM function that returned the unknown result.
        operation: &'static str,
        /// Raw result integer returned by libNVVM.
        code: c_int,
    },
}

/// `libdevice.10.bc` could not be located on this system. `tried` lists
/// every path that was probed, in order, joined by newlines.
#[derive(Debug, Error)]
#[error(
    "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE or a CUDA Toolkit root, or install the CUDA Toolkit. Tried:\n  {tried}"
)]
pub struct LibdeviceNotFound {
    /// Newline-joined list of paths that were probed.
    pub tried: String,
}

// ============================================================================
// Library handle
// ============================================================================

/// Loaded libNVVM library plus resolved function pointers.
///
/// Hold one of these for the lifetime of any [`Program`] that borrows it.
/// `LibNvvm` owns the underlying `dlopen` handle; dropping it unloads the
/// library, which invalidates any function pointers obtained from it.
///
/// It is fine to call [`LibNvvm::load`] more than once if you want
/// independent handles; each call performs its own `dlopen` and resolves
/// its own symbols.
pub struct LibNvvm {
    _lib: Library,
    create_program: unsafe extern "C" fn(*mut NvvmProgram) -> NvvmResultRaw,
    destroy_program: unsafe extern "C" fn(*mut NvvmProgram) -> NvvmResultRaw,
    add_module:
        unsafe extern "C" fn(NvvmProgram, *const c_char, usize, *const c_char) -> NvvmResultRaw,
    compile_program:
        unsafe extern "C" fn(NvvmProgram, c_int, *const *const c_char) -> NvvmResultRaw,
    get_compiled_result_size: unsafe extern "C" fn(NvvmProgram, *mut usize) -> NvvmResultRaw,
    get_compiled_result: unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResultRaw,
    get_program_log_size: unsafe extern "C" fn(NvvmProgram, *mut usize) -> NvvmResultRaw,
    get_program_log: unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResultRaw,
    get_error_string: unsafe extern "C" fn(NvvmResultRaw) -> *const c_char,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvvmResultRaw,
}

// SAFETY: After `load()`, the struct contains only `extern "C"` function
// pointers and an owned `libloading::Library` handle. The function pointers
// are pure values and the library handle is `Send + Sync` (`libloading`
// guarantees this). libNVVM itself is internally synchronized for
// `nvvmProgram` operations on distinct programs, and we never share a single
// `Program` across threads (it does not implement `Send`).
unsafe impl Send for LibNvvm {}
unsafe impl Sync for LibNvvm {}

/// Resolve a symbol to a function pointer of inferred type `T`.
///
/// `T` is inferred from the field assignment context, so each `resolve(...)`
/// call at the [`LibNvvm::load`] site picks up the precise function-pointer
/// type of the field it is assigned to.
///
/// # Safety
///
/// The returned function pointer is valid only while the borrowed `lib`
/// remains loaded. Callers store the resolved pointer in [`LibNvvm`]
/// alongside the owning `Library`, so the pointer's lifetime matches the
/// `LibNvvm` instance.
unsafe fn resolve<T: Copy>(lib: &Library, name: &'static str) -> Result<T, NvvmError> {
    let sym: Symbol<T> =
        unsafe { lib.get(name.as_bytes()) }.map_err(|source| NvvmError::SymbolNotFound {
            symbol: name,
            source,
        })?;
    Ok(unsafe { *sym.into_raw() })
}

impl LibNvvm {
    /// Locate and load `libnvvm.so` at runtime, then resolve every libNVVM
    /// function this crate uses. Returns [`NvvmError::LibraryNotFound`] if
    /// none of the candidate paths could be opened, or
    /// [`NvvmError::SymbolNotFound`] if the loaded library is missing a
    /// required symbol.
    ///
    /// See the crate-level docs for the exact discovery order.
    pub fn load() -> Result<Self, NvvmError> {
        let mut tried = Vec::new();
        let lib = open_library(&mut tried).ok_or_else(|| NvvmError::LibraryNotFound {
            tried: tried.join("\n  "),
        })?;

        unsafe {
            Ok(LibNvvm {
                create_program: resolve(&lib, "nvvmCreateProgram")?,
                destroy_program: resolve(&lib, "nvvmDestroyProgram")?,
                add_module: resolve(&lib, "nvvmAddModuleToProgram")?,
                compile_program: resolve(&lib, "nvvmCompileProgram")?,
                get_compiled_result_size: resolve(&lib, "nvvmGetCompiledResultSize")?,
                get_compiled_result: resolve(&lib, "nvvmGetCompiledResult")?,
                get_program_log_size: resolve(&lib, "nvvmGetProgramLogSize")?,
                get_program_log: resolve(&lib, "nvvmGetProgramLog")?,
                get_error_string: resolve(&lib, "nvvmGetErrorString")?,
                version: resolve(&lib, "nvvmVersion")?,
                _lib: lib,
            })
        }
    }

    /// Query libNVVM's version as `(major, minor)`. Wraps `nvvmVersion`,
    /// which returns the supported NVVM IR version (e.g. CUDA 13's libNVVM
    /// reports `(2, 0)`).
    ///
    /// Returns [`NvvmError::Call`] if the underlying call fails.
    pub fn version(&self) -> Result<(i32, i32), NvvmError> {
        let mut major = 0;
        let mut minor = 0;
        let r = unsafe { (self.version)(&mut major, &mut minor) };
        check(self, r, "nvvmVersion", None)?;
        Ok((major, minor))
    }
}

// ============================================================================
// Program (RAII)
// ============================================================================

/// RAII wrapper around an `nvvmProgram` handle.
///
/// Typical usage:
///
/// 1. [`Program::new`] to create a fresh handle.
/// 2. One or more [`Program::add_module`] calls to feed in NVVM IR text or
///    LLVM bitcode (e.g. `libdevice.10.bc` plus the kernel module).
/// 3. [`Program::compile`] with libNVVM options (`-arch=...`, `-gen-lto`,
///    ...) to produce PTX or LTOIR bytes.
///
/// The handle is destroyed on drop. `Program` borrows the [`LibNvvm`] that
/// created it, so the library outlives every program handle.
pub struct Program<'a> {
    nvvm: &'a LibNvvm,
    handle: NvvmProgram,
}

impl<'a> Program<'a> {
    /// Create a fresh `nvvmProgram` handle. Wraps `nvvmCreateProgram`.
    pub fn new(nvvm: &'a LibNvvm) -> Result<Self, NvvmError> {
        let mut handle = NvvmProgram(ptr::null_mut());
        let r = unsafe { (nvvm.create_program)(&mut handle) };
        check(nvvm, r, "nvvmCreateProgram", None)?;
        Ok(Self { nvvm, handle })
    }

    /// Add an NVVM IR (text) or LLVM bitcode module to the program. Wraps
    /// `nvvmAddModuleToProgram`.
    ///
    /// `name` is recorded by libNVVM for use in diagnostic messages and
    /// program-log output. It does not need to correspond to a file on
    /// disk.
    ///
    /// # Panics
    ///
    /// Panics if `name` contains an interior NUL byte.
    pub fn add_module(&mut self, ir: &[u8], name: &str) -> Result<(), NvvmError> {
        let cname = CString::new(name).expect("module name has interior NUL");
        let r = unsafe {
            (self.nvvm.add_module)(
                self.handle,
                ir.as_ptr() as *const c_char,
                ir.len(),
                cname.as_ptr(),
            )
        };
        let log = self.try_log();
        check(self.nvvm, r, "nvvmAddModuleToProgram", log)
    }

    /// Compile every previously-added module and return the produced PTX or
    /// LTOIR bytes. Wraps `nvvmCompileProgram` + `nvvmGetCompiledResult`.
    ///
    /// `options` are passed to libNVVM verbatim. Common choices:
    /// - `-arch=compute_XY` -- target compute capability (required).
    /// - `-gen-lto` -- emit LTOIR (instead of the default PTX).
    /// - `-opt=3` -- optimization level (`0`–`3`).
    ///
    /// On failure, returns [`NvvmError::Call`] with the libNVVM program log
    /// attached so the original NVVM diagnostic is preserved.
    ///
    /// # Panics
    ///
    /// Panics if any option string contains an interior NUL byte.
    pub fn compile(&mut self, options: &[&str]) -> Result<Vec<u8>, NvvmError> {
        let coptions: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("option has interior NUL"))
            .collect();
        let optr: Vec<*const c_char> = coptions.iter().map(|s| s.as_ptr()).collect();

        let r =
            unsafe { (self.nvvm.compile_program)(self.handle, optr.len() as c_int, optr.as_ptr()) };
        let log = self.try_log();
        check(self.nvvm, r, "nvvmCompileProgram", log)?;

        let mut size: usize = 0;
        let r = unsafe { (self.nvvm.get_compiled_result_size)(self.handle, &mut size) };
        check(self.nvvm, r, "nvvmGetCompiledResultSize", None)?;

        let mut buf = vec![0u8; size];
        let r = unsafe {
            (self.nvvm.get_compiled_result)(self.handle, buf.as_mut_ptr() as *mut c_char)
        };
        check(self.nvvm, r, "nvvmGetCompiledResult", None)?;

        Ok(buf)
    }

    /// Best-effort retrieval of the program log (warnings + errors).
    /// Returns `None` if the log is empty or cannot be fetched.
    fn try_log(&self) -> Option<String> {
        let mut size: usize = 0;
        let r = unsafe { (self.nvvm.get_program_log_size)(self.handle, &mut size) };
        if NvvmResultCode::from_raw(r) != Some(NvvmResultCode::Success) || size <= 1 {
            return None;
        }
        let mut buf = vec![0u8; size];
        let r =
            unsafe { (self.nvvm.get_program_log)(self.handle, buf.as_mut_ptr() as *mut c_char) };
        if NvvmResultCode::from_raw(r) != Some(NvvmResultCode::Success) {
            return None;
        }
        // Trim trailing NUL.
        if let Some(&0) = buf.last() {
            buf.pop();
        }
        Some(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl Drop for Program<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.nvvm.destroy_program)(&mut self.handle);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn check(
    nvvm: &LibNvvm,
    r: NvvmResultRaw,
    op: &'static str,
    log: Option<String>,
) -> Result<(), NvvmError> {
    let code = NvvmResultCode::from_raw(r).ok_or(NvvmError::UnknownResult {
        operation: op,
        code: r,
    })?;
    if code == NvvmResultCode::Success {
        return Ok(());
    }
    Err(NvvmError::Call {
        operation: op,
        code,
        log: log.or_else(|| error_string(nvvm, code)),
    })
}

fn error_string(nvvm: &LibNvvm, r: NvvmResultCode) -> Option<String> {
    let p = unsafe { (nvvm.get_error_string)(r.as_raw()) };
    if p.is_null() {
        return None;
    }
    Some(
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
enum LibraryCandidate {
    Path(PathBuf),
    LoaderName(&'static str),
    SearchPattern {
        dir: Option<PathBuf>,
        pattern: &'static str,
    },
}

impl LibraryCandidate {
    fn description(&self) -> String {
        match self {
            Self::Path(path) => path.display().to_string(),
            Self::LoaderName(name) => (*name).to_string(),
            Self::SearchPattern {
                dir: Some(dir),
                pattern,
            } => dir.join(pattern).display().to_string(),
            Self::SearchPattern { dir: None, pattern } => (*pattern).to_string(),
        }
    }
}

fn open_library(tried: &mut Vec<String>) -> Option<Library> {
    let override_path = std::env::var_os("LIBNVVM_PATH").map(PathBuf::from);
    let discovered = cuda_toolkit_discovery::libnvvm_dll_candidates(target_triple_hint());
    let candidates = library_candidates(override_path, &discovered);
    open_library_from_candidates(&candidates, tried)
}

fn open_library_from_candidates(
    candidates: &[LibraryCandidate],
    tried: &mut Vec<String>,
) -> Option<Library> {
    for candidate in candidates {
        match candidate {
            LibraryCandidate::Path(path) => {
                if let Some(lib) = try_open_path(path, tried) {
                    return Some(lib);
                }
            }
            LibraryCandidate::LoaderName(name) => {
                tried.push((*name).to_string());
                if let Ok(lib) = unsafe { Library::new(*name) } {
                    return Some(lib);
                }
            }
            LibraryCandidate::SearchPattern { dir, pattern } => {
                tried.push(candidate.description());
                for path in matching_files(dir.as_deref(), pattern) {
                    if let Some(lib) = try_open_path(&path, tried) {
                        return Some(lib);
                    }
                }
            }
        }
    }

    None
}

fn try_open_path(path: &Path, tried: &mut Vec<String>) -> Option<Library> {
    tried.push(path.display().to_string());
    unsafe { Library::new(path) }.ok()
}

fn library_candidates(
    override_path: Option<PathBuf>,
    discovered_paths: &[PathBuf],
) -> Vec<LibraryCandidate> {
    let mut candidates = Vec::new();
    if let Some(path) = override_path {
        candidates.push(LibraryCandidate::Path(path));
    }

    platform_library_candidates(&mut candidates, discovered_paths);
    candidates
}

#[cfg(windows)]
fn platform_library_candidates(
    candidates: &mut Vec<LibraryCandidate>,
    discovered_paths: &[PathBuf],
) {
    candidates.push(LibraryCandidate::SearchPattern {
        dir: None,
        pattern: "nvvm64_*.dll",
    });

    for path in discovered_paths {
        if let Some(parent) = path.parent() {
            push_candidate_once(
                candidates,
                LibraryCandidate::SearchPattern {
                    dir: Some(parent.to_path_buf()),
                    pattern: "nvvm64_*.dll",
                },
            );
        }
    }

    for path in discovered_paths {
        push_candidate_once(candidates, LibraryCandidate::Path(path.clone()));
    }
}

#[cfg(not(windows))]
fn platform_library_candidates(
    candidates: &mut Vec<LibraryCandidate>,
    discovered_paths: &[PathBuf],
) {
    for soname in ["libnvvm.so.4", "libnvvm.so.3", "libnvvm.so"] {
        candidates.push(LibraryCandidate::LoaderName(soname));
    }

    for path in discovered_paths {
        push_candidate_once(candidates, LibraryCandidate::Path(path.clone()));
    }
}

fn push_candidate_once(candidates: &mut Vec<LibraryCandidate>, candidate: LibraryCandidate) {
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

fn matching_files(dir: Option<&Path>, pattern: &str) -> Vec<PathBuf> {
    let mut matches = Vec::new();
    let Some((prefix, suffix)) = pattern.split_once('*') else {
        return matches;
    };

    for dir in search_dirs(dir) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(file_name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if wildcard_match(file_name, prefix, suffix) {
                matches.push(path);
            }
        }
    }

    matches.sort_by(|a, b| b.file_name().cmp(&a.file_name()).then_with(|| b.cmp(a)));
    matches.dedup();
    matches
}

fn wildcard_match(file_name: &str, prefix: &str, suffix: &str) -> bool {
    #[cfg(windows)]
    {
        let file_name = file_name.to_ascii_lowercase();
        let prefix = prefix.to_ascii_lowercase();
        let suffix = suffix.to_ascii_lowercase();
        file_name.starts_with(&prefix) && file_name.ends_with(&suffix)
    }

    #[cfg(not(windows))]
    {
        file_name.starts_with(prefix) && file_name.ends_with(suffix)
    }
}

fn search_dirs(dir: Option<&Path>) -> Vec<PathBuf> {
    if let Some(dir) = dir {
        return vec![dir.to_path_buf()];
    }

    let mut dirs = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd);
    }
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    dirs
}

fn target_triple_hint() -> &'static str {
    if cfg!(windows) {
        "x86_64-pc-windows-msvc"
    } else {
        "x86_64-unknown-linux-gnu"
    }
}

// ============================================================================
// libdevice discovery
// ============================================================================

/// Locate `libdevice.10.bc` from the CUDA Toolkit.
///
/// libdevice ships in the toolkit's `nvvm/` component alongside libNVVM and is
/// consumed together with libNVVM in the LTOIR pipeline, so its discovery lives
/// here next to the library discovery in [`LibNvvm::load`].
///
/// Search order:
/// 1. `CUDA_OXIDE_LIBDEVICE` env var (used as-is if it points to an existing
///    file).
/// 2. CUDA Toolkit roots discovered by `cuda-toolkit-discovery`.
///
/// Returns [`LibdeviceNotFound`] with the full list of probed paths if nothing
/// matches.
pub fn find_libdevice() -> Result<PathBuf, LibdeviceNotFound> {
    find_libdevice_with(
        || std::env::var_os("CUDA_OXIDE_LIBDEVICE").map(PathBuf::from),
        cuda_toolkit_discovery::libdevice_candidates,
        |path| path.exists(),
    )
}

fn find_libdevice_with(
    mut override_path: impl FnMut() -> Option<PathBuf>,
    candidates: impl FnOnce() -> Vec<PathBuf>,
    mut exists: impl FnMut(&Path) -> bool,
) -> Result<PathBuf, LibdeviceNotFound> {
    let mut tried = Vec::new();
    if let Some(path) = override_path() {
        tried.push(path.display().to_string());
        if exists(&path) {
            return Ok(path);
        }
    }

    for candidate in candidates() {
        tried.push(candidate.display().to_string());
        if exists(&candidate) {
            return Ok(candidate);
        }
    }

    Err(LibdeviceNotFound {
        tried: tried.join("\n  "),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn candidate_descriptions(override_path: Option<PathBuf>, roots: &[PathBuf]) -> Vec<String> {
        library_candidates(override_path, roots)
            .iter()
            .map(LibraryCandidate::description)
            .collect()
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{now}", std::process::id()))
    }

    #[test]
    fn direct_override_is_first_candidate() {
        let override_path = PathBuf::from(r"C:\custom\nvvm64_40_0.dll");
        let descriptions =
            candidate_descriptions(Some(override_path.clone()), &[PathBuf::from(r"C:\CUDA")]);

        assert_eq!(descriptions[0], override_path.display().to_string());
    }

    #[cfg(windows)]
    #[test]
    fn windows_candidates_include_loader_and_toolkit_patterns() {
        let root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0");
        let discovered = vec![
            root.join("nvvm")
                .join("bin")
                .join("x64")
                .join("nvvm64_40_0.dll"),
        ];
        let descriptions = candidate_descriptions(None, &discovered);

        assert!(descriptions.contains(&"nvvm64_*.dll".to_string()));
        assert!(
            descriptions.contains(
                &root
                    .join("nvvm")
                    .join("bin")
                    .join("x64")
                    .join("nvvm64_*.dll")
                    .display()
                    .to_string()
            )
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn linux_candidates_preserve_loader_names_and_toolkit_path() {
        let root = PathBuf::from("/usr/local/cuda");
        let discovered = vec![root.join("nvvm/lib64/libnvvm.so")];
        let descriptions = candidate_descriptions(None, &discovered);

        assert_eq!(descriptions[0], "libnvvm.so.4");
        assert_eq!(descriptions[1], "libnvvm.so.3");
        assert_eq!(descriptions[2], "libnvvm.so");
        assert!(descriptions.contains(&root.join("nvvm/lib64/libnvvm.so").display().to_string()));
    }

    #[test]
    fn wildcard_scan_finds_versioned_dlls_without_glob_dependency() {
        let dir = unique_temp_dir("libnvvm-sys-dll-scan");
        fs::create_dir_all(&dir).expect("create temp dll scan dir");
        fs::write(dir.join("nvvm64_30_0.dll"), []).expect("write old dll");
        fs::write(dir.join("nvvm64_40_0.dll"), []).expect("write new dll");
        fs::write(dir.join("not-nvvm64_40_0.dll"), []).expect("write nonmatch");

        let matches = matching_files(Some(&dir), "nvvm64_*.dll");
        let names: Vec<_> = matches
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names, ["nvvm64_40_0.dll", "nvvm64_30_0.dll"]);

        fs::remove_dir_all(dir).expect("remove temp dll scan dir");
    }

    #[test]
    fn unknown_result_code_is_not_mapped_to_known_enum() {
        assert_eq!(NvvmResultCode::from_raw(10_000), None);
    }

    #[test]
    fn known_result_code_displays_symbolic_name() {
        assert_eq!(
            NvvmResultCode::from_raw(7).unwrap().to_string(),
            "InvalidOption"
        );
    }

    #[test]
    fn find_libdevice_honors_explicit_override_file() {
        let found = find_libdevice_with(
            || Some(PathBuf::from("/elsewhere/libdevice.10.bc")),
            Vec::new,
            |path| path == Path::new("/elsewhere/libdevice.10.bc"),
        );

        assert_eq!(found.unwrap(), PathBuf::from("/elsewhere/libdevice.10.bc"));
    }

    #[test]
    fn find_libdevice_probes_candidates_in_order() {
        let found = find_libdevice_with(
            || None,
            || {
                vec![
                    PathBuf::from("/cuda/first/nvvm/libdevice/libdevice.10.bc"),
                    PathBuf::from("/cuda/second/nvvm/libdevice/libdevice.10.bc"),
                ]
            },
            |path| path == Path::new("/cuda/second/nvvm/libdevice/libdevice.10.bc"),
        );

        assert_eq!(
            found.unwrap(),
            PathBuf::from("/cuda/second/nvvm/libdevice/libdevice.10.bc")
        );
    }

    #[test]
    fn find_libdevice_failure_lists_every_probed_path() {
        let err = find_libdevice_with(
            || Some(PathBuf::from("/override/libdevice.10.bc")),
            || vec![PathBuf::from("/cuda/nvvm/libdevice/libdevice.10.bc")],
            |_| false,
        )
        .unwrap_err();

        assert_eq!(
            err.tried,
            "/override/libdevice.10.bc\n  /cuda/nvvm/libdevice/libdevice.10.bc"
        );
        let message = err.to_string();
        assert!(message.contains("CUDA_OXIDE_LIBDEVICE"));
        assert!(message.contains("CUDA Toolkit root"));
    }

    #[test]
    #[ignore = "requires a CUDA Toolkit libNVVM library available to the loader"]
    fn smoke_load_libnvvm_version() {
        let nvvm = LibNvvm::load().expect("load libNVVM");
        let _version = nvvm.version().expect("query libNVVM version");
    }
}
