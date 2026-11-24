fn main() {
    #[cfg(feature = "cuda")]
    {
        cc::Build::new()
            .cuda(true)
            .file("src/cuda/kernels.cu")
            .flag("-O3")
            .compile("p2_nnue_cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    }
}
