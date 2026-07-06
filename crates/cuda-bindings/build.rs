/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_toolkit_discovery::{cuda_driver_lib_candidates, include_candidates_for_target};
use std::{env, error::Error, path::Path, path::PathBuf, process::exit};

/// Runs [`run`]; on error, prints the message and exits with status 1.
fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

/// Configures the crate build: declares rerun triggers, discovers the CUDA
/// include directory, adds native link search paths for `libcuda`, links
/// `cuda`, and invokes bindgen on `wrapper.h` with the discovered include
/// directory, writing `bindings.rs` into `OUT_DIR`.
fn run() -> Result<(), Box<dyn Error>> {
    let target = env::var("TARGET").unwrap_or_default();
    if target.ends_with("windows-gnu") {
        return Err(std::io::Error::other(
            "cuda-oxide Windows support currently targets x86_64-pc-windows-msvc. Use the MSVC toolchain.",
        )
        .into());
    }

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_PATH_V13_3");
    println!("cargo::rustc-check-cfg=cfg(cuda_has_cuEventElapsedTime_v2)");

    let include_dir = find_cuda_include_dir(&target)?;
    probe_event_elapsed_time_v2(&include_dir.join("cuda.h"));

    for path in cuda_driver_lib_candidates(&target) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    println!("cargo:rustc-link-lib=dylib=cuda");

    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // CUDA 13.2+ adds types to CUlaunchAttributeValue that bindgen/libclang
        // cannot translate, collapsing the struct to a 1-byte opaque blob while the
        // size assertion still expects the real C size. Making both the struct and its
        // inner union opaque produces correctly-sized byte blobs across CUDA versions.
        // launch_kernel_ex in cuda-core constructs this struct via raw pointer writes.
        .opaque_type("CUlaunchAttribute_st")
        .opaque_type("CUlaunchAttributeValue_union")
        .generate()
        .map_err(|error| format!("cuda-bindings: failed to generate CUDA bindings: {error}"))?
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

/// Returns the include directory containing `cuda.h`.
///
/// A missing `cuda.h` is a hard error here: bindgen cannot run without it,
/// and failing now produces one clean message instead of raw clang diagnostics.
fn find_cuda_include_dir(target: &str) -> Result<PathBuf, String> {
    let candidates = include_candidates_for_target(target);
    if let Some(dir) = candidates.iter().find(|dir| dir.join("cuda.h").is_file()) {
        return Ok(dir.clone());
    }

    let probed: Vec<String> = candidates
        .iter()
        .map(|dir| format!("  {}", dir.join("cuda.h").display()))
        .collect();
    Err(format!(
        "cuda-bindings: could not find cuda.h in the CUDA toolkit.\n\
         Probed:\n\
         {}\n\
         Set CUDA_TOOLKIT_PATH, CUDA_HOME, CUDA_PATH, or CUDA_PATH_V* \
         to a CUDA Toolkit install root.",
        probed.join("\n")
    ))
}

/// Probes the discovered `cuda.h` for `cuEventElapsedTime_v2` and emits the
/// `cuda_has_cuEventElapsedTime_v2` cfg when present.
///
/// CUDA 12.8 renamed the event elapsed-time driver entry point to
/// `cuEventElapsedTime_v2`; earlier toolkits only declare
/// `cuEventElapsedTime`. The cfg lets `src/lib.rs` dispatch to whichever
/// symbol the headers used for this build actually declare.
///
/// A missing `cuda.h` is already a hard error in [`find_cuda_include_dir`];
/// a present but unreadable `cuda.h` stays a warning here (treated as the
/// pre-12.8 spelling) because bindgen reports the authoritative failure
/// right after.
fn probe_event_elapsed_time_v2(cuda_h: &Path) {
    println!("cargo:rerun-if-changed={}", cuda_h.display());
    match std::fs::read_to_string(cuda_h) {
        Ok(header) => {
            if header.contains("cuEventElapsedTime_v2") {
                println!("cargo:rustc-cfg=cuda_has_cuEventElapsedTime_v2");
            }
        }
        Err(error) => {
            println!(
                "cargo:warning=cuda-bindings: failed to probe {}: {error}",
                cuda_h.display()
            );
        }
    }
}
