/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import <Foundation/Foundation.h>

/**
 * Distance metric used for vector comparisons.
 * Integer values match the C PistaDBMetric enum.
 */
typedef NS_ENUM(NSInteger, PSTMetric) {
    /** Euclidean distance – general purpose, image/multimodal embeddings. */
    PSTMetricL2      = 0,
    /** Cosine similarity stored as 1 − similarity – text embeddings.     */
    PSTMetricCosine  = 1,
    /** Inner product stored as negative dot product – normalised vectors.*/
    PSTMetricIP      = 2,
    /** Manhattan / L1 distance – sparse vectors.                         */
    PSTMetricL1      = 3,
    /** Hamming distance – binary embeddings, deduplication.              */
    PSTMetricHamming = 4,
} NS_SWIFT_NAME(PSTMetric);
