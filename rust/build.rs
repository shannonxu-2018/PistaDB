//! Build script — locates and links the native pistadb C library.
//!
//! Search order:
//!   1. `PISTADB_LIB_DIR` environment variable (highest priority)
//!   2. `../build`, `../build/Release`, `../build/Debug`  (relative to this crate)

fn main() {
    println!("cargo:rerun-if-env-changed=PISTADB_LIB_DIR");

    if let Ok(dir) = std::env::var("PISTADB_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir);
    } else {
        // Crate is at <repo>/rust/ — walk up one level to reach the repo root.
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let root = std::path::Path::new(&manifest_dir)
            .parent()
            .expect("could not resolve repo root from CARGO_MANIFEST_DIR");

        for sub in &["build", "build/Release", "build/Debug"] {
            println!("cargo:rustc-link-search=native={}", root.join(sub).display());
        }
    }

    // Dynamic link against pistadb.
    // Runtime resolves: pistadb.dll (Windows) | libpistadb.so (Linux) | libpistadb.dylib (macOS)
    println!("cargo:rustc-link-lib=dylib=pistadb");

    // Linux requires explicit math library.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=m");
    }
}
