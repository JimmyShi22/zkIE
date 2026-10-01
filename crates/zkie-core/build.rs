//! Compiles the Goldilocks CUDA kernels (Sppark NTT wrapper + Poseidon2
//! Merkle kernels) with nvcc and archives them with Sppark's `all_gpus`
//! device helpers into a static library.
//!
//! Mirrors the BN254 MSM crate's pattern: CUDA 12.4 is the tested toolchain
//! (`CUDA_NVCC=/usr/local/cuda-12.4/bin/nvcc`, `CUDA_HOME=/usr/local/cuda-12.4`),
//! and the `sppark` crate must be built with `NVCC=off` so it doesn't compile
//! its own copy of `util/all_gpus.cpp`.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_goldilocks();

    println!("cargo:rerun-if-env-changed=CUDA_NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=cuda");
}

#[cfg(feature = "cuda")]
fn add_includes(cmd: &mut Command, sppark_root: &str) {
    // Local kernels + the Sppark C++ headers (`<sppark/...>`, `<util/...>`).
    cmd.arg("-I").arg("cuda");
    cmd.arg("-I").arg(sppark_root);
}

#[cfg(feature = "cuda")]
fn compile_cuda_goldilocks() {
    let nvcc = env::var("CUDA_NVCC")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "nvcc".to_string());

    if which::which(&nvcc).is_err() {
        panic!(
            "CUDA backend enabled but nvcc not found. Set CUDA_NVCC=/path/to/nvcc (tested with CUDA 12.4)."
        );
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let sppark_root = env::var("DEP_SPPARK_ROOT").expect("DEP_SPPARK_ROOT not set");

    // NTT (purpose-built radix-2 DIT over gl64_t).
    let ntt_obj = out_dir.join("goldilocks_ntt.o");
    let mut c1 = Command::new(&nvcc);
    c1.arg("-c")
        .arg("-O3")
        .arg("-arch=sm_89")
        .arg("-t0")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-Xcompiler")
        .arg("-Wno-unused-function");
    add_includes(&mut c1, &sppark_root);
    c1.arg("-o").arg(&ntt_obj).arg("cuda/goldilocks_ntt.cu");
    let status = c1.status().expect("failed to spawn nvcc");
    if !status.success() {
        panic!("nvcc failed to compile cuda/goldilocks_ntt.cu");
    }

    // Poseidon2 + Merkle tree kernels.
    let p2_obj = out_dir.join("poseidon2_merkle.o");
    let mut c2 = Command::new(&nvcc);
    c2.arg("-c")
        .arg("-O3")
        .arg("-arch=sm_89")
        .arg("-t0")
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-Xcompiler")
        .arg("-Wno-unused-function");
    add_includes(&mut c2, &sppark_root);
    c2.arg("-o").arg(&p2_obj).arg("cuda/poseidon2_merkle.cu");
    let status = c2.status().expect("failed to spawn nvcc");
    if !status.success() {
        panic!("nvcc failed to compile cuda/poseidon2_merkle.cu");
    }

    // Host-side GPU helpers used by the kernels (gpu_props, cuda_available, ...).
    let all_gpus_obj = out_dir.join("all_gpus.o");
    let mut c3 = Command::new(&nvcc);
    c3.arg("-c").arg("-O3").arg("-Xcompiler").arg("-fPIC");
    add_includes(&mut c3, &sppark_root);
    c3.arg("-o")
        .arg(&all_gpus_obj)
        .arg(PathBuf::from(&sppark_root).join("util/all_gpus.cpp"));
    let status = c3.status().expect("failed to spawn nvcc");
    if !status.success() {
        panic!("nvcc failed to compile util/all_gpus.cpp");
    }

    let lib = out_dir.join("libzkie_goldilocks_cuda.a");
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = Command::new(ar)
        .arg("rcs")
        .arg(&lib)
        .arg(&ntt_obj)
        .arg(&p2_obj)
        .arg(&all_gpus_obj)
        .status()
        .expect("failed to spawn ar");
    if !status.success() {
        panic!("ar failed to archive CUDA objects");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=zkie_goldilocks_cuda");

    if let Some(cuda_home) = env::var_os("CUDA_HOME") {
        println!(
            "cargo:rustc-link-search=native={}",
            PathBuf::from(cuda_home).join("lib64").display()
        );
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gcc_s");
}
