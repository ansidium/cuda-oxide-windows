/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared CUDA Toolkit discovery helpers for build scripts and runtime loaders.
//!
//! These helpers return candidate paths in discovery order. They do not require
//! the CUDA Toolkit, a driver, or a GPU to be present.

use std::{
    cmp::Ordering,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

const ROOT_ENV_VARS: &[&str] = &["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"];
const WINDOWS_CUDA_DEFAULT_ROOT: &str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
const WINDOWS_CUDA_DEFAULT_VERSIONS: &[&str] = &[
    "v13.3", "v13.2", "v13.1", "v13.0", "v12.9", "v12.8", "v12.6", "v12.5", "v12.4", "v12.3",
    "v12.2", "v12.1", "v12.0", "v11.8", "v11.7", "v11.6", "v11.5", "v11.4", "v11.3", "v11.2",
    "v11.1", "v11.0",
];
const LINUX_CUDA_DEFAULT_ROOTS: &[&str] = &["/usr/local/cuda", "/opt/cuda"];

/// Candidate CUDA Toolkit include directories, in root discovery order.
pub fn include_candidates() -> Vec<PathBuf> {
    include_candidates_from_roots(root_candidates(DefaultRoots::All))
}

/// Candidate CUDA Toolkit include directories for a target triple.
pub fn include_candidates_for_target(target: &str) -> Vec<PathBuf> {
    include_candidates_from_roots(root_candidates(DefaultRoots::for_target(target)))
}

/// Candidate CUDA Toolkit roots, independent of any specific library kind.
pub fn cuda_roots() -> Vec<PathBuf> {
    root_candidates(DefaultRoots::All)
}

/// Candidate native library search directories for the CUDA driver library.
pub fn cuda_driver_lib_candidates(target: &str) -> Vec<PathBuf> {
    let roots = root_candidates(DefaultRoots::for_target(target));
    if is_windows_target(target) {
        dedup(
            roots
                .into_iter()
                .map(|root| root.join("lib").join("x64"))
                .collect(),
        )
    } else {
        dedup(
            roots
                .into_iter()
                .flat_map(|root| {
                    [
                        root.join("lib64"),
                        root.join("lib64").join("stubs"),
                        root.join("targets").join("x86_64-linux").join("lib"),
                        root.join("targets")
                            .join("x86_64-linux")
                            .join("lib")
                            .join("stubs"),
                    ]
                })
                .collect(),
        )
    }
}

/// Candidate paths to the libNVVM dynamic library.
pub fn libnvvm_dll_candidates(target: &str) -> Vec<PathBuf> {
    let roots = root_candidates(DefaultRoots::for_target(target));
    if is_windows_target(target) {
        windows_runtime_library_candidates(&roots, &["nvvm64_40_0.dll"], |root| {
            root.join("nvvm").join("bin").join("x64")
        })
    } else {
        dedup(
            roots
                .into_iter()
                .map(|root| root.join("nvvm").join("lib64").join("libnvvm.so"))
                .collect(),
        )
    }
}

/// Candidate paths to the nvJitLink dynamic library.
pub fn nvjitlink_dll_candidates(target: &str) -> Vec<PathBuf> {
    let roots = root_candidates(DefaultRoots::for_target(target));
    if is_windows_target(target) {
        windows_runtime_library_candidates(
            &roots,
            &["nvJitLink_130_0.dll", "nvJitLink_120_0.dll"],
            |root| root.join("bin").join("x64"),
        )
    } else {
        dedup(
            roots
                .into_iter()
                .map(|root| root.join("lib64").join("libnvJitLink.so"))
                .collect(),
        )
    }
}

/// Candidate paths to CUDA libdevice bitcode.
pub fn libdevice_candidates() -> Vec<PathBuf> {
    dedup(
        root_candidates(DefaultRoots::All)
            .into_iter()
            .map(|root| root.join("nvvm").join("libdevice").join("libdevice.10.bc"))
            .collect(),
    )
}

/// Runtime directories that may need to be appended to the process search path.
pub fn path_dirs_for_runtime(target: &str) -> Vec<PathBuf> {
    let roots = root_candidates(DefaultRoots::for_target(target));
    if is_windows_target(target) {
        dedup(
            roots
                .into_iter()
                .flat_map(|root| {
                    [
                        root.join("bin"),
                        root.join("bin").join("x64"),
                        root.join("nvvm").join("bin").join("x64"),
                    ]
                })
                .collect(),
        )
    } else {
        dedup(
            roots
                .into_iter()
                .flat_map(|root| [root.join("lib64"), root.join("nvvm").join("lib64")])
                .collect(),
        )
    }
}

#[derive(Clone, Copy)]
enum DefaultRoots {
    All,
    Linux,
    WindowsThenLinux,
}

impl DefaultRoots {
    fn for_target(target: &str) -> Self {
        if is_windows_target(target) {
            Self::WindowsThenLinux
        } else {
            Self::Linux
        }
    }
}

fn root_candidates(defaults: DefaultRoots) -> Vec<PathBuf> {
    root_candidates_from_env(std::env::vars_os(), defaults)
}

fn root_candidates_from_env<I>(env: I, defaults: DefaultRoots) -> Vec<PathBuf>
where
    I: IntoIterator<Item = (OsString, OsString)>,
{
    let entries = env
        .into_iter()
        .filter_map(|(key, value)| key.into_string().ok().map(|key| (key, value)))
        .collect::<Vec<_>>();
    let mut roots = Vec::new();

    for key in ROOT_ENV_VARS {
        if let Some(value) = env_value(&entries, key) {
            push_if_not_empty(&mut roots, value);
        }
    }

    let mut versioned = entries
        .iter()
        .filter(|(key, value)| key.starts_with("CUDA_PATH_V") && !value.is_empty())
        .collect::<Vec<_>>();
    versioned.sort_by(|(left_key, _), (right_key, _)| {
        compare_cuda_path_version_vars(right_key, left_key).then_with(|| left_key.cmp(right_key))
    });
    for (_, value) in versioned {
        push_if_not_empty(&mut roots, value);
    }

    match defaults {
        DefaultRoots::All | DefaultRoots::WindowsThenLinux => {
            roots.extend(windows_default_roots());
        }
        DefaultRoots::Linux => {}
    }
    match defaults {
        DefaultRoots::All | DefaultRoots::WindowsThenLinux | DefaultRoots::Linux => {
            roots.extend(LINUX_CUDA_DEFAULT_ROOTS.iter().map(PathBuf::from));
        }
    }

    dedup(roots)
}

fn include_candidates_from_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    dedup(roots.into_iter().map(|root| root.join("include")).collect())
}

fn env_value<'a>(entries: &'a [(String, OsString)], key: &str) -> Option<&'a OsStr> {
    entries
        .iter()
        .find_map(|(entry_key, value)| (entry_key == key).then_some(value.as_os_str()))
}

fn push_if_not_empty(roots: &mut Vec<PathBuf>, value: &OsStr) {
    if !value.is_empty() {
        roots.push(PathBuf::from(value));
    }
}

fn windows_default_roots() -> Vec<PathBuf> {
    let base = PathBuf::from(WINDOWS_CUDA_DEFAULT_ROOT);
    let mut roots = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && looks_like_cuda_version_dir(&path) {
                roots.push(path);
            }
        }
        roots.sort_by(|left, right| compare_cuda_version_paths(right, left));
    }

    roots.extend(
        WINDOWS_CUDA_DEFAULT_VERSIONS
            .iter()
            .map(|version| base.join(version)),
    );
    roots
}

fn windows_runtime_library_candidates<F>(
    roots: &[PathBuf],
    fallback_names: &[&str],
    dir_for_root: F,
) -> Vec<PathBuf>
where
    F: Fn(&Path) -> PathBuf,
{
    let mut candidates = Vec::new();
    for root in roots {
        let dir = dir_for_root(root);
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut files = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .and_then(|ext| ext.to_str())
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("dll"))
                })
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            fallback_names.iter().any(|fallback| {
                                name.eq_ignore_ascii_case(fallback)
                                    || name_prefix_before_version(name)
                                        == name_prefix_before_version(fallback)
                            })
                        })
                })
                .collect::<Vec<_>>();
            files.sort();
            candidates.extend(files);
        }
        candidates.extend(fallback_names.iter().map(|name| dir.join(name)));
    }
    dedup(candidates)
}

fn looks_like_cuda_version_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.strip_prefix('v')
                .is_some_and(|version| version.split('.').all(|part| part.parse::<u16>().is_ok()))
        })
}

fn compare_cuda_version_paths(left: &Path, right: &Path) -> Ordering {
    version_parts(
        left.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default(),
    )
    .cmp(&version_parts(
        right
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default(),
    ))
}

fn compare_cuda_path_version_vars(left: &str, right: &str) -> Ordering {
    version_parts(left).cmp(&version_parts(right))
}

fn version_parts(value: &str) -> Vec<u16> {
    value
        .split(|ch: char| !ch.is_ascii_digit())
        .filter(|part| !part.is_empty())
        .filter_map(|part| part.parse().ok())
        .collect()
}

fn name_prefix_before_version(name: &str) -> &str {
    name.find(|ch: char| ch.is_ascii_digit())
        .map_or(name, |index| &name[..index])
}

fn is_windows_target(target: &str) -> bool {
    target.contains("windows")
}

fn dedup(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for path in paths {
        if !deduped.contains(&path) {
            deduped.push(path);
        }
    }
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOWS_TARGET: &str = "x86_64-pc-windows-msvc";
    const LINUX_TARGET: &str = "x86_64-unknown-linux-gnu";

    #[test]
    fn windows_default_v13_3_candidates_are_present() {
        let root = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3");
        let roots = root_candidates_from_env(Vec::<(OsString, OsString)>::new(), DefaultRoots::All);

        assert!(roots.contains(&root));
        assert!(include_candidates_from_roots(roots.clone()).contains(&root.join("include")));
        assert!(
            cuda_driver_lib_candidates_from_roots(roots.clone(), WINDOWS_TARGET)
                .contains(&root.join("lib").join("x64"))
        );
        assert!(
            path_dirs_for_runtime_from_roots(roots.clone(), WINDOWS_TARGET)
                .contains(&root.join("bin"))
        );
        assert!(
            path_dirs_for_runtime_from_roots(roots.clone(), WINDOWS_TARGET)
                .contains(&root.join("bin").join("x64"))
        );
        assert!(
            path_dirs_for_runtime_from_roots(roots.clone(), WINDOWS_TARGET)
                .contains(&root.join("nvvm").join("bin").join("x64"))
        );
        assert!(
            libdevice_candidates_from_roots(roots)
                .contains(&root.join("nvvm").join("libdevice").join("libdevice.10.bc"))
        );
    }

    #[test]
    fn linux_usr_local_cuda_candidates_are_present() {
        let root = PathBuf::from("/usr/local/cuda");
        let roots =
            root_candidates_from_env(Vec::<(OsString, OsString)>::new(), DefaultRoots::Linux);

        assert!(roots.contains(&root));
        assert!(include_candidates_from_roots(roots.clone()).contains(&root.join("include")));
        assert!(
            cuda_driver_lib_candidates_from_roots(roots.clone(), LINUX_TARGET)
                .contains(&root.join("lib64"))
        );
        assert!(
            cuda_driver_lib_candidates_from_roots(roots.clone(), LINUX_TARGET)
                .contains(&root.join("lib64").join("stubs"))
        );
        assert!(
            libnvvm_dll_candidates_from_roots(roots.clone(), LINUX_TARGET)
                .contains(&root.join("nvvm").join("lib64").join("libnvvm.so"))
        );
        assert!(
            nvjitlink_dll_candidates_from_roots(roots.clone(), LINUX_TARGET)
                .contains(&root.join("lib64").join("libnvJitLink.so"))
        );
        assert!(
            libdevice_candidates_from_roots(roots)
                .contains(&root.join("nvvm").join("libdevice").join("libdevice.10.bc"))
        );
    }

    #[test]
    fn cuda_path_only_is_first_root() {
        let cuda_path = OsString::from(r"D:\NVIDIA\CUDA\v13.3");
        let roots = root_candidates_from_env(
            vec![(OsString::from("CUDA_PATH"), cuda_path.clone())],
            DefaultRoots::All,
        );

        assert_eq!(roots.first(), Some(&PathBuf::from(cuda_path)));
        assert_eq!(
            include_candidates_from_roots(roots).first(),
            Some(&PathBuf::from(r"D:\NVIDIA\CUDA\v13.3").join("include"))
        );
    }

    #[test]
    fn cuda_toolkit_path_precedes_cuda_path() {
        let roots = root_candidates_from_env(
            vec![
                (
                    OsString::from("CUDA_PATH"),
                    OsString::from(r"D:\CUDA\from-cuda-path"),
                ),
                (
                    OsString::from("CUDA_TOOLKIT_PATH"),
                    OsString::from(r"D:\CUDA\from-toolkit-path"),
                ),
            ],
            DefaultRoots::All,
        );

        assert_eq!(
            roots.first(),
            Some(&PathBuf::from(r"D:\CUDA\from-toolkit-path"))
        );
        assert_eq!(
            roots.get(1),
            Some(&PathBuf::from(r"D:\CUDA\from-cuda-path"))
        );
    }

    #[test]
    fn spaces_in_cuda_root_are_preserved() {
        let root = PathBuf::from(r"D:\CUDA Toolkit Installs\v13.3");
        let roots = root_candidates_from_env(
            vec![(OsString::from("CUDA_PATH"), root.clone().into_os_string())],
            DefaultRoots::All,
        );

        assert!(include_candidates_from_roots(roots.clone()).contains(&root.join("include")));
        assert!(
            cuda_driver_lib_candidates_from_roots(roots.clone(), WINDOWS_TARGET)
                .contains(&root.join("lib").join("x64"))
        );
        assert!(
            libdevice_candidates_from_roots(roots)
                .contains(&root.join("nvvm").join("libdevice").join("libdevice.10.bc"))
        );
    }

    fn cuda_driver_lib_candidates_from_roots(roots: Vec<PathBuf>, target: &str) -> Vec<PathBuf> {
        if is_windows_target(target) {
            dedup(
                roots
                    .into_iter()
                    .map(|root| root.join("lib").join("x64"))
                    .collect(),
            )
        } else {
            dedup(
                roots
                    .into_iter()
                    .flat_map(|root| {
                        [
                            root.join("lib64"),
                            root.join("lib64").join("stubs"),
                            root.join("targets").join("x86_64-linux").join("lib"),
                            root.join("targets")
                                .join("x86_64-linux")
                                .join("lib")
                                .join("stubs"),
                        ]
                    })
                    .collect(),
            )
        }
    }

    fn libnvvm_dll_candidates_from_roots(roots: Vec<PathBuf>, target: &str) -> Vec<PathBuf> {
        if is_windows_target(target) {
            windows_runtime_library_candidates(&roots, &["nvvm64_40_0.dll"], |root| {
                root.join("nvvm").join("bin").join("x64")
            })
        } else {
            dedup(
                roots
                    .into_iter()
                    .map(|root| root.join("nvvm").join("lib64").join("libnvvm.so"))
                    .collect(),
            )
        }
    }

    fn nvjitlink_dll_candidates_from_roots(roots: Vec<PathBuf>, target: &str) -> Vec<PathBuf> {
        if is_windows_target(target) {
            windows_runtime_library_candidates(
                &roots,
                &["nvJitLink_130_0.dll", "nvJitLink_120_0.dll"],
                |root| root.join("bin").join("x64"),
            )
        } else {
            dedup(
                roots
                    .into_iter()
                    .map(|root| root.join("lib64").join("libnvJitLink.so"))
                    .collect(),
            )
        }
    }

    fn libdevice_candidates_from_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
        dedup(
            roots
                .into_iter()
                .map(|root| root.join("nvvm").join("libdevice").join("libdevice.10.bc"))
                .collect(),
        )
    }

    fn path_dirs_for_runtime_from_roots(roots: Vec<PathBuf>, target: &str) -> Vec<PathBuf> {
        if is_windows_target(target) {
            dedup(
                roots
                    .into_iter()
                    .flat_map(|root| {
                        [
                            root.join("bin"),
                            root.join("bin").join("x64"),
                            root.join("nvvm").join("bin").join("x64"),
                        ]
                    })
                    .collect(),
            )
        } else {
            dedup(
                roots
                    .into_iter()
                    .flat_map(|root| [root.join("lib64"), root.join("nvvm").join("lib64")])
                    .collect(),
            )
        }
    }
}
