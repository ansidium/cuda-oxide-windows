/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime (`dlopen`) bindings to NVIDIA's nvJitLink.
//!
//! nvJitLink links one or more LTOIR modules (and other input forms) into
//! a final cubin or PTX. It is part of the CUDA Toolkit and ships at
//! `<cuda>/lib64/libnvJitLink.so` on Linux and as `nvJitLink_*.dll` under
//! `<cuda>/bin/` or `<cuda>/bin/x64/` on Windows.
//!
//! # Library discovery
//!
//! [`LibNvJitLink::load`] tries (in order):
//! 1. `LIBNVJITLINK_PATH` env var, if set.
//! 2. Platform loader names (`libnvJitLink.so.13`, `libnvJitLink.so.12`,
//!    `libnvJitLink.so` on Linux; discovered `nvJitLink_*.dll` files on
//!    Windows).
//! 3. CUDA Toolkit roots from `cuda-toolkit-discovery`, including
//!    `<root>/lib64/libnvJitLink.so` on Linux and
//!    `<root>/bin/x64/nvJitLink_*.dll` / `<root>/bin/nvJitLink_*.dll` on
//!    Windows.
//!
//! # Symbol naming
//!
//! `nvJitLink.h` `#define`s every public function to a versioned mangled
//! name, e.g. `nvJitLinkCreate -> __nvJitLinkCreate_13_0`, but the library
//! also exports the unversioned name with default ELF symbol versioning.
//! That means `dlsym(handle, "nvJitLinkCreate")` resolves to the right
//! function on every CUDA Toolkit version, so this binding does not need
//! to probe per-CUDA-version symbol suffixes.
//!
//! # Example
//!
//! ```no_run
//! use nvjitlink_sys::{LibNvJitLink, Linker, InputType};
//!
//! let nvj = LibNvJitLink::load().expect("CUDA Toolkit (nvJitLink) not found");
//! let mut linker = Linker::new(&nvj, &["-arch=sm_120", "-lto"]).unwrap();
//! let ltoir = std::fs::read("kernel.ltoir").unwrap();
//! linker.add(InputType::Ltoir, &ltoir, "kernel.ltoir").unwrap();
//! let cubin = linker.finish().unwrap();
//! ```

use libloading::{Library, Symbol};
use std::ffi::{CString, c_char, c_int, c_void};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::SystemTime;
use thiserror::Error;

// ============================================================================
// FFI types
// ============================================================================

/// Opaque nvJitLink handle (`nvJitLinkHandle`).
#[repr(transparent)]
#[derive(Copy, Clone)]
struct NvJitLinkHandle(*mut c_void);

/// Integer representation of nvJitLink's C `nvJitLinkResult` enum.
///
/// This is an integer rather than a Rust enum so result codes added by newer
/// nvJitLink versions remain valid values.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct NvJitLinkResult(c_int);

impl NvJitLinkResult {
    const SUCCESS: Self = Self(0);
}

/// nvJitLink input kinds (`nvJitLinkInputType`). Mirrors `nvJitLink.h`.
///
/// Pass to [`Linker::add`] to tell nvJitLink how to interpret a chunk of
/// input bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InputType {
    /// Sentinel "no input" value. Not a valid argument to [`Linker::add`].
    None = 0,
    /// CUDA binary (cubin).
    Cubin = 1,
    /// PTX assembly.
    Ptx = 2,
    /// LTOIR — the output of libNVVM `compile(... "-gen-lto" ...)`.
    Ltoir = 3,
    /// CUDA fat binary.
    Fatbin = 4,
    /// Host object file.
    Object = 5,
    /// Host library archive.
    Library = 6,
    /// Index file (used with sliced fatbins).
    Index = 7,
    /// Auto-detect the kind from the bytes. Convenient but slower; prefer
    /// the specific variant when you know the input format.
    Any = 10,
}

// ============================================================================
// Errors
// ============================================================================

/// All errors surfaced by this crate.
#[derive(Debug, Error)]
pub enum NvJitLinkError {
    /// nvJitLink could not be located on this system. `tried` lists every
    /// path, loader name, or search pattern that was probed, in order.
    #[error(
        "nvJitLink could not be located. Set LIBNVJITLINK_PATH or a CUDA Toolkit root, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibraryNotFound {
        /// Newline-joined list of paths, loader names, and search patterns.
        tried: String,
    },

    /// nvJitLink was loaded, but symbol lookup failed to resolve a function
    /// this crate requires. Indicates an old or broken nvJitLink that does
    /// not export the standard linker API.
    #[error("nvJitLink was found but a required symbol is missing: {symbol}: {source}")]
    SymbolNotFound {
        /// Name of the missing nvJitLink function (e.g. `nvJitLinkCreate`).
        symbol: &'static str,
        /// Underlying `libloading` error returned by `dlsym`.
        #[source]
        source: libloading::Error,
    },

    /// The loaded nvJitLink predates linked-PTX retrieval. Cubin output is
    /// still usable; only callers that explicitly request PTX receive this
    /// error.
    #[error(
        "the loaded libnvJitLink does not export nvJitLinkGetLinkedPtxSize/nvJitLinkGetLinkedPtx"
    )]
    PtxOutputUnavailable,

    /// An nvJitLink call returned a non-`Success` `nvJitLinkResult`. `log`
    /// carries the nvJitLink error log when one was produced by the call.
    #[error("nvJitLink error in {operation}: {code:?}{}", .log.as_ref().map(|l| format!("\n--- nvJitLink error log ---\n{l}")).unwrap_or_default())]
    Call {
        /// Name of the nvJitLink function that failed.
        operation: &'static str,
        /// Raw `nvJitLinkResult` integer.
        code: i32,
        /// nvJitLink error log, if available.
        log: Option<String>,
    },
}

// ============================================================================
// Library handle
// ============================================================================

/// Loaded nvJitLink library plus resolved function pointers.
///
/// Hold one of these for the lifetime of any [`Linker`] that borrows it.
/// `LibNvJitLink` owns the underlying `dlopen` handle; dropping it unloads
/// the library, which invalidates any function pointers obtained from it.
///
/// It is fine to call [`LibNvJitLink::load`] more than once if you want
/// independent handles; each call performs its own `dlopen` and resolves
/// its own symbols.
pub struct LibNvJitLink {
    _lib: Library,
    loaded_file: Option<File>,
    loaded_identity: Option<LibraryFileIdentity>,
    create:
        unsafe extern "C" fn(*mut NvJitLinkHandle, u32, *const *const c_char) -> NvJitLinkResult,
    destroy: unsafe extern "C" fn(*mut NvJitLinkHandle) -> NvJitLinkResult,
    add_data: unsafe extern "C" fn(
        NvJitLinkHandle,
        InputType,
        *const c_void,
        usize,
        *const c_char,
    ) -> NvJitLinkResult,
    complete: unsafe extern "C" fn(NvJitLinkHandle) -> NvJitLinkResult,
    get_linked_cubin_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_linked_cubin: unsafe extern "C" fn(NvJitLinkHandle, *mut c_void) -> NvJitLinkResult,
    get_linked_ptx_size:
        Option<unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult>,
    get_linked_ptx: Option<unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult>,
    get_error_log_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_error_log: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
    get_info_log_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_info_log: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
    version: Option<unsafe extern "C" fn(*mut u32, *mut u32) -> NvJitLinkResult>,
}

// SAFETY: Same reasoning as `libnvvm-sys::LibNvvm`. The struct holds an
// owned `libloading::Library` (which is `Send + Sync`) and a set of
// `extern "C"` function pointers. We never share a single `Linker` across
// threads (it is not `Send`), so per-handle thread safety is not required
// from nvJitLink itself.
unsafe impl Send for LibNvJitLink {}
unsafe impl Sync for LibNvJitLink {}

/// Resolve a required symbol to a function pointer of inferred type `T`.
///
/// # Safety
///
/// The returned function pointer is valid only while the borrowed `lib`
/// remains loaded. Callers store the resolved pointer in [`LibNvJitLink`]
/// alongside the owning `Library`, so the pointer's lifetime matches the
/// `LibNvJitLink` instance.
unsafe fn resolve<T: Copy>(lib: &Library, name: &'static str) -> Result<T, NvJitLinkError> {
    let sym: Symbol<T> =
        unsafe { lib.get(name.as_bytes()) }.map_err(|source| NvJitLinkError::SymbolNotFound {
            symbol: name,
            source,
        })?;
    Ok(unsafe { *sym.into_raw() })
}

/// Resolve an optional symbol; returns `None` if missing.
///
/// Used for symbols that may not be present on older CUDA Toolkit versions
/// (e.g. `nvJitLinkVersion`, added in CTK 12.3).
///
/// # Safety
///
/// Same as [`resolve`].
unsafe fn resolve_optional<T: Copy>(lib: &Library, name: &'static str) -> Option<T> {
    let sym: Symbol<T> = unsafe { lib.get(name.as_bytes()) }.ok()?;
    Some(unsafe { *sym.into_raw() })
}

impl LibNvJitLink {
    /// Locate and load `libnvJitLink.so` at runtime, then resolve every
    /// nvJitLink function this crate uses.
    ///
    /// Returns [`NvJitLinkError::LibraryNotFound`] if none of the candidate
    /// paths could be opened, or [`NvJitLinkError::SymbolNotFound`] if the
    /// loaded library is missing a required symbol. See the crate-level
    /// docs for the exact discovery order.
    pub fn load() -> Result<Self, NvJitLinkError> {
        Self::load_inner(false)
    }

    /// Load nvJitLink while retaining an exact, fingerprintable descriptor
    /// when the platform supports it.
    ///
    /// This is intended for a process-wide pinned linker cache handle. On
    /// Linux it opens the concrete library before `dlopen` and retains that
    /// descriptor so callers can fingerprint the selected file. Callers must
    /// retain the returned `LibNvJitLink` for the process lifetime and restart
    /// to change toolkits. General callers should use [`LibNvJitLink::load`].
    #[doc(hidden)]
    pub fn load_for_cache() -> Result<Self, NvJitLinkError> {
        Self::load_inner(true)
    }

    fn load_inner(retain_exact_file: bool) -> Result<Self, NvJitLinkError> {
        let mut tried = Vec::new();
        let opened = open_library(&mut tried, retain_exact_file).ok_or_else(|| {
            NvJitLinkError::LibraryNotFound {
                tried: tried.join("\n  "),
            }
        })?;
        let OpenedLibrary {
            library: lib,
            loaded_file,
            loaded_identity,
        } = opened;

        unsafe {
            Ok(LibNvJitLink {
                create: resolve(&lib, "nvJitLinkCreate")?,
                destroy: resolve(&lib, "nvJitLinkDestroy")?,
                add_data: resolve(&lib, "nvJitLinkAddData")?,
                complete: resolve(&lib, "nvJitLinkComplete")?,
                get_linked_cubin_size: resolve(&lib, "nvJitLinkGetLinkedCubinSize")?,
                get_linked_cubin: resolve(&lib, "nvJitLinkGetLinkedCubin")?,
                // These symbols are optional so older toolkits continue to
                // support cubin output.
                get_linked_ptx_size: resolve_optional(&lib, "nvJitLinkGetLinkedPtxSize"),
                get_linked_ptx: resolve_optional(&lib, "nvJitLinkGetLinkedPtx"),
                get_error_log_size: resolve(&lib, "nvJitLinkGetErrorLogSize")?,
                get_error_log: resolve(&lib, "nvJitLinkGetErrorLog")?,
                get_info_log_size: resolve(&lib, "nvJitLinkGetInfoLogSize")?,
                get_info_log: resolve(&lib, "nvJitLinkGetInfoLog")?,
                version: resolve_optional(&lib, "nvJitLinkVersion"),
                loaded_file,
                loaded_identity,
                _lib: lib,
            })
        }
    }

    /// Return the exact file descriptor used to load nvJitLink, provided that
    /// its contents have not changed since `dlopen`.
    ///
    /// [`LibNvJitLink::load_for_cache`] opens concrete library paths before
    /// loading them and retains the descriptor. Callers may fingerprint it to
    /// bind cached linker output to the process-pinned tool. Ordinary
    /// [`LibNvJitLink::load`] calls return `None` here. Any `None` result means
    /// cache reuse must be skipped.
    #[doc(hidden)]
    pub fn loaded_file_if_unchanged(&self) -> Option<&File> {
        let identity = self.loaded_identity.as_ref()?;
        let file = self.loaded_file.as_ref()?;
        identity.matches_file(file).then_some(file)
    }

    /// Query nvJitLink's version as `(major, minor)`. Wraps
    /// `nvJitLinkVersion` (added in CTK 12.3).
    ///
    /// Returns `None` if the loaded library does not export
    /// `nvJitLinkVersion`, or if the call itself fails.
    pub fn version(&self) -> Option<(u32, u32)> {
        let f = self.version?;
        let mut major = 0;
        let mut minor = 0;
        let r = unsafe { f(&mut major, &mut minor) };
        if r == NvJitLinkResult::SUCCESS {
            Some((major, minor))
        } else {
            None
        }
    }
}

// ============================================================================
// Linker (RAII)
// ============================================================================

/// RAII wrapper around an `nvJitLinkHandle`.
///
/// Typical usage:
///
/// 1. [`Linker::new`] with the link options (`-arch=sm_XX`, `-lto`, ...).
/// 2. One or more [`Linker::add`] calls feeding LTOIR / PTX / cubin chunks.
/// 3. [`Linker::finish`] to drive the link and return the cubin bytes.
///
/// The handle is destroyed on drop. `Linker` borrows the [`LibNvJitLink`]
/// that created it, so the library outlives every linker handle.
pub struct Linker<'a> {
    nvj: &'a LibNvJitLink,
    handle: NvJitLinkHandle,
}

impl<'a> Linker<'a> {
    /// Create a fresh linker. Wraps `nvJitLinkCreate`.
    ///
    /// `options` are passed to nvJitLink verbatim. Common choices:
    /// - `-arch=sm_XY` -- target SM (required).
    /// - `-lto` -- enable link-time optimization (required to consume
    ///   LTOIR inputs).
    /// - `-time` / `-verbose` -- emit timing or info messages into the
    ///   nvJitLink info log.
    ///
    /// # Panics
    ///
    /// Panics if any option string contains an interior NUL byte.
    pub fn new(nvj: &'a LibNvJitLink, options: &[&str]) -> Result<Self, NvJitLinkError> {
        let coptions: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("option has interior NUL"))
            .collect();
        let optr: Vec<*const c_char> = coptions.iter().map(|s| s.as_ptr()).collect();

        let mut handle = NvJitLinkHandle(ptr::null_mut());
        let r = unsafe { (nvj.create)(&mut handle, optr.len() as u32, optr.as_ptr()) };
        check(
            nvj,
            &Linker {
                nvj,
                handle: NvJitLinkHandle(ptr::null_mut()),
            },
            r,
            "nvJitLinkCreate",
        )?;
        Ok(Self { nvj, handle })
    }

    /// Add a single input chunk (in `kind` format) to the link. Wraps
    /// `nvJitLinkAddData`.
    ///
    /// `name` is recorded by nvJitLink for use in diagnostic messages and
    /// info-log output. It does not need to correspond to a file on disk.
    ///
    /// # Panics
    ///
    /// Panics if `name` contains an interior NUL byte.
    pub fn add(&mut self, kind: InputType, data: &[u8], name: &str) -> Result<(), NvJitLinkError> {
        let cname = CString::new(name).expect("input name has interior NUL");
        let r = unsafe {
            (self.nvj.add_data)(
                self.handle,
                kind,
                data.as_ptr() as *const c_void,
                data.len(),
                cname.as_ptr(),
            )
        };
        check(self.nvj, self, r, "nvJitLinkAddData")
    }

    /// Drive the link and return the resulting cubin bytes. Wraps
    /// `nvJitLinkComplete` + `nvJitLinkGetLinkedCubin`.
    ///
    /// Consumes the [`Linker`]; on success the underlying handle is freed
    /// after the cubin has been copied out. On failure, the cubin is empty
    /// and the [`NvJitLinkError::Call`] carries the nvJitLink error log.
    ///
    /// If `CUDA_OXIDE_VERBOSE` is set in the environment, the nvJitLink
    /// info log (timings, sm_XY chosen, etc.) is forwarded to `stderr`.
    pub fn finish(self) -> Result<Vec<u8>, NvJitLinkError> {
        let r = unsafe { (self.nvj.complete)(self.handle) };
        check(self.nvj, &self, r, "nvJitLinkComplete")?;

        let mut size: usize = 0;
        let r = unsafe { (self.nvj.get_linked_cubin_size)(self.handle, &mut size) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedCubinSize")?;

        let mut buf = vec![0u8; size];
        let r =
            unsafe { (self.nvj.get_linked_cubin)(self.handle, buf.as_mut_ptr() as *mut c_void) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedCubin")?;

        // Forward the info log if anyone is listening (helpful with `-verbose`).
        if let Some(info) = self.try_info_log()
            && std::env::var_os("CUDA_OXIDE_VERBOSE").is_some()
        {
            eprintln!("--- nvJitLink info log ---\n{info}");
        }

        Ok(buf)
    }

    /// Drive the link and return linked PTX text.
    ///
    /// Construct the linker with both `-lto` and `-ptx`. Unlike
    /// [`Self::finish`], this retrieves `nvJitLinkGetLinkedPtx*` output. The
    /// returned buffer may include nvJitLink's trailing NUL byte, which is
    /// accepted by the CUDA driver and useful for direct `cuModuleLoadData`.
    ///
    /// The PTX functions are optional so older nvJitLink versions can still
    /// produce cubins.
    pub fn finish_ptx(self) -> Result<Vec<u8>, NvJitLinkError> {
        let get_size = self
            .nvj
            .get_linked_ptx_size
            .ok_or(NvJitLinkError::PtxOutputUnavailable)?;
        let get = self
            .nvj
            .get_linked_ptx
            .ok_or(NvJitLinkError::PtxOutputUnavailable)?;

        let r = unsafe { (self.nvj.complete)(self.handle) };
        check(self.nvj, &self, r, "nvJitLinkComplete")?;

        let mut size = 0;
        let r = unsafe { get_size(self.handle, &mut size) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedPtxSize")?;

        let mut buf = vec![0u8; size];
        let r = unsafe { get(self.handle, buf.as_mut_ptr() as *mut c_char) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedPtx")?;

        if let Some(info) = self.try_info_log()
            && std::env::var_os("CUDA_OXIDE_VERBOSE").is_some()
        {
            eprintln!("--- nvJitLink info log ---\n{info}");
        }

        Ok(buf)
    }

    /// Best-effort retrieval of the error log.
    fn try_error_log(&self) -> Option<String> {
        try_log(
            self.nvj,
            self.handle,
            self.nvj.get_error_log_size,
            self.nvj.get_error_log,
        )
    }

    /// Best-effort retrieval of the info log.
    fn try_info_log(&self) -> Option<String> {
        try_log(
            self.nvj,
            self.handle,
            self.nvj.get_info_log_size,
            self.nvj.get_info_log,
        )
    }
}

impl Drop for Linker<'_> {
    fn drop(&mut self) {
        if !self.handle.0.is_null() {
            unsafe {
                (self.nvj.destroy)(&mut self.handle);
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn check(
    _nvj: &LibNvJitLink,
    linker: &Linker<'_>,
    r: NvJitLinkResult,
    op: &'static str,
) -> Result<(), NvJitLinkError> {
    if r == NvJitLinkResult::SUCCESS {
        return Ok(());
    }
    Err(NvJitLinkError::Call {
        operation: op,
        code: r.0,
        log: linker.try_error_log(),
    })
}

fn try_log(
    _nvj: &LibNvJitLink,
    handle: NvJitLinkHandle,
    size_fn: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_fn: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
) -> Option<String> {
    if handle.0.is_null() {
        return None;
    }
    let mut size: usize = 0;
    let r = unsafe { size_fn(handle, &mut size) };
    if r != NvJitLinkResult::SUCCESS || size <= 1 {
        return None;
    }
    let mut buf = vec![0u8; size];
    let r = unsafe { get_fn(handle, buf.as_mut_ptr() as *mut c_char) };
    if r != NvJitLinkResult::SUCCESS {
        return None;
    }
    if let Some(&0) = buf.last() {
        buf.pop();
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

#[derive(Debug, PartialEq, Eq)]
struct LibraryFileIdentity {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_time: (i64, i64),
}

impl LibraryFileIdentity {
    fn capture_file(file: &File) -> Option<Self> {
        Self::from_metadata(&file.metadata().ok()?)
    }

    fn from_metadata(metadata: &std::fs::Metadata) -> Option<Self> {
        let modified = metadata.modified().ok()?;

        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        Some(Self {
            len: metadata.len(),
            modified,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            change_time: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }

    fn matches_file(&self, file: &File) -> bool {
        Self::capture_file(file).as_ref() == Some(self)
    }

    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fn matches_path(&self, path: &Path) -> bool {
        path.metadata()
            .ok()
            .as_ref()
            .and_then(Self::from_metadata)
            .as_ref()
            == Some(self)
    }
}

struct OpenedLibrary {
    library: Library,
    loaded_file: Option<File>,
    loaded_identity: Option<LibraryFileIdentity>,
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

fn open_library(tried: &mut Vec<String>, retain_exact_file: bool) -> Option<OpenedLibrary> {
    let override_path = std::env::var_os("LIBNVJITLINK_PATH").map(PathBuf::from);
    let discovered = cuda_toolkit_discovery::nvjitlink_dll_candidates(target_triple_hint());
    let candidates = library_candidates(override_path, &discovered);
    open_library_from_candidates(&candidates, tried, retain_exact_file)
}

fn open_library_from_candidates(
    candidates: &[LibraryCandidate],
    tried: &mut Vec<String>,
    retain_exact_file: bool,
) -> Option<OpenedLibrary> {
    for candidate in candidates {
        match candidate {
            LibraryCandidate::Path(path) => {
                tried.push(path.display().to_string());
                if let Some(opened) = open_library_path(path, retain_exact_file) {
                    return Some(opened);
                }
            }
            LibraryCandidate::LoaderName(name) => {
                tried.push((*name).to_string());
                if let Ok(lib) = unsafe { Library::new(*name) } {
                    return Some(OpenedLibrary {
                        library: lib,
                        loaded_file: None,
                        loaded_identity: None,
                    });
                }
            }
            LibraryCandidate::SearchPattern { dir, pattern } => {
                tried.push(candidate.description());
                for path in matching_files(dir.as_deref(), pattern) {
                    tried.push(path.display().to_string());
                    if let Some(opened) = open_library_path(&path, retain_exact_file) {
                        return Some(opened);
                    }
                }
            }
        }
    }

    None
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
        pattern: "nvJitLink_*.dll",
    });

    for path in discovered_paths {
        let Some(parent) = path.parent() else {
            continue;
        };
        push_candidate_once(
            candidates,
            LibraryCandidate::SearchPattern {
                dir: Some(parent.to_path_buf()),
                pattern: "nvJitLink_*.dll",
            },
        );

        if parent
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case("x64"))
            && let Some(bin_dir) = parent.parent()
        {
            push_candidate_once(
                candidates,
                LibraryCandidate::SearchPattern {
                    dir: Some(bin_dir.to_path_buf()),
                    pattern: "nvJitLink_*.dll",
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
    for soname in [
        "libnvJitLink.so.13",
        "libnvJitLink.so.12",
        "libnvJitLink.so",
    ] {
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

fn open_library_path(path: &Path, retain_exact_file: bool) -> Option<OpenedLibrary> {
    #[cfg(not(target_os = "linux"))]
    let _ = retain_exact_file;
    #[cfg(target_os = "linux")]
    let canonical_path = path.canonicalize().ok();

    #[cfg(target_os = "linux")]
    if retain_exact_file
        && let Some(canonical_path) = canonical_path.as_deref()
        && let Ok(file) = File::open(canonical_path)
        && file.metadata().is_ok_and(|metadata| metadata.is_file())
    {
        let identity = LibraryFileIdentity::capture_file(&file);
        // Load the same absolute file we opened. Re-resolving a relative path
        // could select another DSO if the process working directory changes.
        if let Ok(lib) = unsafe { Library::new(canonical_path) } {
            let identity = identity.filter(|identity| {
                identity.matches_file(&file) && identity.matches_path(canonical_path)
            });
            return Some(OpenedLibrary {
                library: lib,
                loaded_file: Some(file),
                loaded_identity: identity,
            });
        }
    }

    let lib = unsafe { Library::new(path) }.ok()?;
    Some(OpenedLibrary {
        library: lib,
        // Loading by pathname cannot prove which mapping the dynamic loader
        // returned when another handle already exists for that pathname.
        loaded_file: None,
        loaded_identity: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn candidate_descriptions(
        override_path: Option<PathBuf>,
        discovered_paths: &[PathBuf],
    ) -> Vec<String> {
        library_candidates(override_path, discovered_paths)
            .iter()
            .map(LibraryCandidate::description)
            .collect()
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{now}", std::process::id()))
    }

    #[test]
    fn direct_override_is_first_candidate() {
        let override_path = PathBuf::from(r"C:\custom\nvJitLink_130_0.dll");
        let descriptions =
            candidate_descriptions(Some(override_path.clone()), &[PathBuf::from(r"C:\CUDA")]);

        assert_eq!(descriptions[0], override_path.display().to_string());
    }

    #[cfg(windows)]
    #[test]
    fn windows_candidates_include_loader_and_toolkit_patterns() {
        let root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0");
        let discovered = vec![root.join("bin").join("x64").join("nvJitLink_130_0.dll")];
        let descriptions = candidate_descriptions(None, &discovered);

        assert!(descriptions.contains(&"nvJitLink_*.dll".to_string()));
        assert!(
            descriptions.contains(
                &root
                    .join("bin")
                    .join("x64")
                    .join("nvJitLink_*.dll")
                    .display()
                    .to_string()
            )
        );
        assert!(
            descriptions.contains(
                &root
                    .join("bin")
                    .join("nvJitLink_*.dll")
                    .display()
                    .to_string()
            )
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn linux_candidates_preserve_loader_names_and_toolkit_path() {
        let root = PathBuf::from("/usr/local/cuda");
        let discovered = vec![root.join("lib64/libnvJitLink.so")];
        let descriptions = candidate_descriptions(None, &discovered);

        assert_eq!(descriptions[0], "libnvJitLink.so.13");
        assert_eq!(descriptions[1], "libnvJitLink.so.12");
        assert_eq!(descriptions[2], "libnvJitLink.so");
        assert!(descriptions.contains(&root.join("lib64/libnvJitLink.so").display().to_string()));
    }

    #[test]
    fn wildcard_scan_finds_versioned_dlls_without_glob_dependency() {
        let dir = unique_temp_dir("nvjitlink-sys-dll-scan");
        fs::create_dir_all(&dir).expect("create temp dll scan dir");
        fs::write(dir.join("nvJitLink_120_0.dll"), []).expect("write old dll");
        fs::write(dir.join("nvJitLink_130_0.dll"), []).expect("write new dll");
        fs::write(dir.join("not-nvJitLink_130_0.dll"), []).expect("write nonmatch");

        let matches = matching_files(Some(&dir), "nvJitLink_*.dll");
        let names: Vec<_> = matches
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names, ["nvJitLink_130_0.dll", "nvJitLink_120_0.dll"]);

        fs::remove_dir_all(dir).expect("remove temp dll scan dir");
    }

    #[test]
    fn open_descriptor_remains_bound_to_replaced_inode() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "nvjitlink-sys-identity-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let library_path = directory.join("libnvJitLink.so");
        let replacement_path = directory.join("replacement.so");
        std::fs::write(&library_path, b"original-library").unwrap();
        std::fs::write(
            &replacement_path,
            b"replacement-library-with-different-length",
        )
        .unwrap();

        let canonical_path = library_path.canonicalize().unwrap();
        let opened = File::open(&canonical_path).unwrap();
        let opened_identity = LibraryFileIdentity::capture_file(&opened).unwrap();
        assert!(opened_identity.matches_file(&opened));
        assert!(opened_identity.matches_path(&canonical_path));

        std::fs::remove_file(&library_path).unwrap();
        std::fs::rename(&replacement_path, &library_path).unwrap();
        assert!(!opened_identity.matches_path(&canonical_path));

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            let opened_metadata = opened.metadata().unwrap();
            assert_eq!(opened_identity.device, opened_metadata.dev());
            assert_eq!(opened_identity.inode, opened_metadata.ino());
            let replacement_file = File::open(&canonical_path).unwrap();
            let replacement = LibraryFileIdentity::capture_file(&replacement_file).unwrap();
            assert_ne!(
                (opened_identity.device, opened_identity.inode),
                (replacement.device, replacement.inode)
            );
        }

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn result_representation_accepts_future_error_codes() {
        let future_code = NvJitLinkResult(c_int::MAX);
        assert_ne!(future_code, NvJitLinkResult::SUCCESS);
        assert_eq!(future_code.0, c_int::MAX);
    }

    #[test]
    #[ignore = "requires an installed CUDA Toolkit with nvJitLink"]
    fn installed_toolkit_exposes_linked_ptx_output() {
        let library = LibNvJitLink::load().expect("load nvJitLink");
        assert!(library.get_linked_ptx_size.is_some());
        assert!(library.get_linked_ptx.is_some());
    }
}
