/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * A stored vector and its label, returned by -[PSTDatabase entryForId:error:].
 *
 * The raw float bytes are exposed through @c vectorData (NSData) and the
 * convenience accessors @c count / @c floatAtIndex:.
 * The Swift wrapper converts this directly to @c [Float].
 */
NS_SWIFT_NAME(PSTVectorEntry)
@interface PSTVectorEntry : NSObject

/**
 * Raw float32 bytes (length == dim * sizeof(float)).
 * Bind with @c withUnsafeBytes(_:) in Swift or read via @c floatAtIndex:.
 */
@property (nonatomic, readonly) NSData   *vectorData;

/** Human-readable label supplied at insert time. Empty string if none. */
@property (nonatomic, readonly) NSString *label;

/** Number of float elements (== dim). */
@property (nonatomic, readonly) NSInteger count;

/** Returns the float at the given index (unchecked – caller must validate). */
- (float)floatAtIndex:(NSInteger)index;

- (instancetype)initWithFloats:(const float *)floats
                         count:(NSInteger)count
                         label:(NSString *)label NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
