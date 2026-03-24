/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import <Foundation/Foundation.h>

/**
 * Index algorithm used to organise and search vectors.
 * Integer values match the C PistaDBIndexType enum.
 */
typedef NS_ENUM(NSInteger, PSTIndexType) {
    /** Brute-force linear scan – exact results, no training needed.          */
    PSTIndexTypeLinear  = 0,
    /**
     * Hierarchical Navigable Small World graphs.
     * Best speed / recall trade-off for most use-cases (recommended default).
     */
    PSTIndexTypeHNSW    = 1,
    /**
     * Inverted File Index with k-means clustering.
     * Good for large datasets; -[PSTDatabase train:] required before inserts.
     */
    PSTIndexTypeIVF     = 2,
    /**
     * IVF + Product Quantisation – memory-efficient compression.
     * -[PSTDatabase train:] required before inserts.
     */
    PSTIndexTypeIVFPQ   = 3,
    /** Vamana / DiskANN graphs – optimised for billion-scale datasets.       */
    PSTIndexTypeDiskANN = 4,
    /** Locality-Sensitive Hashing – ultra-low memory footprint.              */
    PSTIndexTypeLSH     = 5,
    /**
     * ScaNN: Anisotropic Vector Quantisation (Google ICML 2020).
     * Highest recall on cosine / IP metrics.
     * -[PSTDatabase train:] required before inserts.
     */
    PSTIndexTypeScaNN   = 6,
} NS_SWIFT_NAME(PSTIndexType);
