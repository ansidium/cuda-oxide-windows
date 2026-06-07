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

/// Configures the crate build: declares rerun triggers, adds native link search paths for `libcuda`,
/// links `cuda`, and invokes bindgen on `wrapper.h` with the discovered CUDA include directory,
/// writing `bindings.rs` into `OUT_DIR`.
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

    let include_dir = select_include_dir(&target);
    let cuda_h = include_dir.join("cuda.h");
    println!("cargo:rerun-if-changed={}", cuda_h.display());

    match std::fs::read_to_string(&cuda_h) {
        Ok(contents) => {
            if contents.contains("cuEventElapsedTime_v2") {
                println!("cargo:rustc-cfg=cuda_has_cuEventElapsedTime_v2");
            }
        }
        Err(err) => {
            println!(
                "cargo:warning=cuda-bindings: Could not read cuda.h at {}: {}",
                cuda_h.display(),
                err
            );
        }
    }

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
        .expect("Unable to generate CUDA bindings")
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

fn select_include_dir(target: &str) -> PathBuf {
    let candidates = include_candidates_for_target(target);
    candidates
        .iter()
        .find(|candidate| candidate.join("cuda.h").is_file())
        .cloned()
        .or_else(|| candidates.first().cloned())
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda").join("include"))
}
