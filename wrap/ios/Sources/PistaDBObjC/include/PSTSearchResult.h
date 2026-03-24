/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * A single k-NN search result returned by -[PSTDatabase searchFloatArray:count:k:error:].
 * Results are ordered by ascending distance (nearest first).
 */
NS_SWIFT_NAME(PSTSearchResult)
@interface PSTSearchResult : NSObject

/** The vector's unique identifier. */
@property (nonatomic, readonly) uint64_t  vectorId;

/**
 * Distance from the query vector (interpretation depends on the metric):
 *  - L2 / L1 / Hamming: lower is closer.
 *  - Cosine: 1 − cosine_similarity, lower is closer.
 *  - IP: negative dot product, lower is closer.
 */
@property (nonatomic, readonly) float     distance;

/** Human-readable label supplied at insert time. Empty string if none. */
@property (nonatomic, readonly) NSString *label;

- (instancetype)initWithId:(uint64_t)vectorId
                  distance:(float)distance
                     label:(NSString *)label NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
