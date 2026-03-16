// swift-tools-version: 5.8
//    ___ _    _        ___  ___
//   | _ (_)__| |_ __ _|   \| _ )
//   |  _/ (_-<  _/ _` | |) | _ \
//   |_| |_/__/\__\__,_|___/|___/

import PackageDescription

let package = Package(
    name: "PistaDB",
    platforms: [
        .iOS(.v13),
        .macOS(.v11),
        .watchOS(.v6),
        .tvOS(.v13),
    ],
    products: [
        // Full Swift API (recommended for Swift projects)
        .library(name: "PistaDB",      targets: ["PistaDB"]),
        // Objective-C API only (for ObjC-only targets)
        .library(name: "PistaDBObjC",  targets: ["PistaDBObjC"]),
    ],
    targets: [

        // ── C core library ──────────────────────────────────────────────────
        // Compiles all 11 .c source files in src/ directly into the package.
        // publicHeadersPath: "." exposes every .h file under src/ to
        // dependents as  #include <CPistaDB/pistadb.h>  etc.
        .target(
            name: "CPistaDB",
            path: "src",
            publicHeadersPath: ".",
            cSettings: [
                .define("_CRT_SECURE_NO_WARNINGS"),
                // Optimisation flags – remove unsafeFlags if your toolchain
                // enforces strict flag validation.
                .unsafeFlags(["-O3", "-ffast-math"],
                             .when(configuration: .release)),
            ]
        ),

        // ── Objective-C wrapper ─────────────────────────────────────────────
        .target(
            name: "PistaDBObjC",
            dependencies: ["CPistaDB"],
            path: "ios/Sources/PistaDBObjC",
            publicHeadersPath: "include",
            cSettings: [
                // Silence ObjC nullability / deprecation noise from the C headers
                .unsafeFlags(["-Wno-nullability-completeness"]),
            ]
        ),

        // ── Swift wrapper ───────────────────────────────────────────────────
        .target(
            name: "PistaDB",
            dependencies: ["PistaDBObjC"],
            path: "ios/Sources/PistaDB"
        ),
    ]
)
