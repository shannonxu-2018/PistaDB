/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import <Foundation/Foundation.h>
#import "PSTError.h"
#import "PSTMetric.h"
#import "PSTIndexType.h"
#import "PSTParams.h"
#import "PSTSearchResult.h"
#import "PSTVectorEntry.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * PSTDatabase – Objective-C interface to the PistaDB embedded vector database.
 *
 * Thread safety: all public methods are @c @synchronized on the receiver.
 * Concurrent reads and writes from multiple threads are serialised.
 *
 * @code
 * NSError *error;
 * PSTDatabase *db = [PSTDatabase databaseWithPath:path
 *                                             dim:384
 *                                          metric:PSTMetricCosine
 *                                       indexType:PSTIndexTypeHNSW
 *                                          params:nil
 *                                           error:&error];
 *
 * [db insertId:1 floatArray:embedding count:384 label:@"doc-1" error:&error];
 *
 * NSArray<PSTSearchResult *> *hits =
 *     [db searchFloatArray:queryVec count:384 k:5 error:&error];
 * @endcode
 */
NS_SWIFT_NAME(PSTDatabase)
@interface PSTDatabase : NSObject

/**
 * Convenience factory.  Returns nil (and populates error) if the database
 * cannot be opened or created.
 */
+ (nullable instancetype)databaseWithPath:(NSString *)path
                                      dim:(NSInteger)dim
                                   metric:(PSTMetric)metric
                                indexType:(PSTIndexType)indexType
                                   params:(nullable PSTParams *)params
                                    error:(NSError *__autoreleasing _Nullable *)error
    NS_SWIFT_NAME(init(path:dim:metric:indexType:params:));

/**
 * Opens or creates a database file.
 *
 * @param path      Path to the @c .pst file.  Created if it does not exist.
 * @param dim       Vector dimensionality (must match any existing file).
 * @param metric    Distance metric.
 * @param indexType Index algorithm.
 * @param params    Tuning parameters.  Pass @c nil for defaults.
 * @param error     On failure, contains a @c PSTDatabaseErrorDomain error.
 */
- (nullable instancetype)initWithPath:(NSString *)path
                                  dim:(NSInteger)dim
                               metric:(PSTMetric)metric
                            indexType:(PSTIndexType)indexType
                               params:(nullable PSTParams *)params
                                error:(NSError *__autoreleasing _Nullable *)error
    NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/* ── Lifecycle ────────────────────────────────────────────────────────────── */

/**
 * Persists the database to disk.
 * Does NOT close the handle; call @c close when finished.
 */
- (BOOL)save:(NSError *__autoreleasing _Nullable *)error;

/**
 * Saves the database then frees the native handle.
 * Safe to call multiple times.
 */
- (void)close;

/* ── CRUD ─────────────────────────────────────────────────────────────────── */

/**
 * Inserts a vector.
 *
 * @param vectorId  Unique identifier (user-managed, must be unique).
 * @param floats    Float array of length @p count.
 * @param count     Must equal @c dim.
 * @param label     Optional human-readable label (max 255 bytes); pass @c nil for none.
 * @param error     Populated on failure.
 * @return @c YES on success.
 */
- (BOOL)insertId:(uint64_t)vectorId
      floatArray:(const float *)floats
           count:(NSInteger)count
           label:(nullable NSString *)label
           error:(NSError *__autoreleasing _Nullable *)error
    NS_SWIFT_NAME(insert(id:floatArray:count:label:));

/**
 * Soft-deletes a vector.  Space is reclaimed on the next @c save.
 *
 * @param vectorId  Identifier supplied at insert time.
 * @param error     Populated on failure.
 * @return @c YES on success.
 */
- (BOOL)deleteId:(uint64_t)vectorId
           error:(NSError *__autoreleasing _Nullable *)error
    NS_SWIFT_NAME(delete(id:));

/**
 * Replaces the vector data for an existing identifier.
 *
 * @param vectorId  Identifier to update.
 * @param floats    Replacement float array of length @p count.
 * @param count     Must equal @c dim.
 * @param error     Populated on failure.
 * @return @c YES on success.
 */
- (BOOL)updateId:(uint64_t)vectorId
      floatArray:(const float *)floats
           count:(NSInteger)count
           error:(NSError *__autoreleasing _Nullable *)error
    NS_SWIFT_NAME(update(id:floatArray:count:));

/**
 * Retrieves a stored vector and its label.
 *
 * @param vectorId  Identifier supplied at insert time.
 * @param error     Populated on failure.
 * @return A @c PSTVectorEntry, or @c nil if not found.
 */
- (nullable PSTVectorEntry *)entryForId:(uint64_t)vectorId
                                  error:(NSError *__autoreleasing _Nullable *)error
    NS_SWIFT_NAME(entry(id:));

/* ── Search ───────────────────────────────────────────────────────────────── */

/**
 * K-nearest-neighbour search.
 *
 * @param query  Float array of length @p count (must equal @c dim).
 * @param count  Must equal @c dim.
 * @param k      Number of results requested.
 * @param error  Populated on failure.
 * @return Up to @p k results ordered by ascending distance, or @c nil on error.
 */
- (nullable NSArray<PSTSearchResult *> *)searchFloatArray:(const float *)query
                                                    count:(NSInteger)count
                                                        k:(NSInteger)k
                                                    error:(NSError *__autoreleasing _Nullable *)error
    NS_SWIFT_NAME(search(floatArray:count:k:));

/* ── Index management ─────────────────────────────────────────────────────── */

/**
 * Trains the index on currently inserted vectors.
 * Required for IVF, IVF_PQ, and ScaNN before inserting.
 * Optional for HNSW / DiskANN (triggers a rebuild pass).
 */
- (BOOL)train:(NSError *__autoreleasing _Nullable *)error;

/* ── Metadata ─────────────────────────────────────────────────────────────── */

/** Number of active (non-deleted) vectors. */
@property (nonatomic, readonly) NSInteger    count;
/** Vector dimensionality. */
@property (nonatomic, readonly) NSInteger    dim;
/** Distance metric in use. */
@property (nonatomic, readonly) PSTMetric    metric;
/** Index algorithm in use. */
@property (nonatomic, readonly) PSTIndexType indexType;
/** Human-readable description of the last error. */
@property (nonatomic, readonly) NSString    *lastError;

/** Library version string, e.g. @c "1.0.0". */
+ (NSString *)version;

@end

NS_ASSUME_NONNULL_END
