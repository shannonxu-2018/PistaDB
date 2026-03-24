/*
 *    ___ _    _        ___  ___
 *   | _ (_)__| |_ __ _|   \| _ )
 *   |  _/ (_-<  _/ _` | |) | _ \
 *   |_| |_/__/\__\__,_|___/|___/
 */
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Index tuning parameters passed when opening or creating a database.
 *
 * All defaults match the C pistadb_default_params() values.
 *
 * Objective-C:
 * @code
 * PSTParams *p = [PSTParams defaultParams];
 * p.hnswM = 32;
 * p.hnswEfSearch = 100;
 * @endcode
 *
 * Swift (prefer the Swift PistaDBParams DSL in the PistaDB module instead):
 * @code
 * let p = PSTParams.default()
 * p.hnswM = 32
 * @endcode
 */
NS_SWIFT_NAME(PSTParams)
@interface PSTParams : NSObject <NSCopying>

/* ── HNSW ─────────────────────────────────────────────────────────────────── */

/** Max connections per layer (default 16). */
@property (nonatomic) NSInteger hnswM;
/** Build-time search width; higher → better recall, slower build (default 200). */
@property (nonatomic) NSInteger hnswEfConstruction;
/** Query-time search width; higher → better recall, slower query (default 50). */
@property (nonatomic) NSInteger hnswEfSearch;

/* ── IVF / IVF_PQ ──────────────────────────────────────────────────────────── */

/** Number of centroids (default 128). */
@property (nonatomic) NSInteger ivfNlist;
/** Centroids to probe at search time (default 8). */
@property (nonatomic) NSInteger ivfNprobe;
/** PQ subspaces – must divide dim evenly (default 8). */
@property (nonatomic) NSInteger pqM;
/** Bits per sub-code: 4 or 8 (default 8). */
@property (nonatomic) NSInteger pqNbits;

/* ── DiskANN ────────────────────────────────────────────────────────────────── */

/** Max graph degree (default 32). */
@property (nonatomic) NSInteger diskannR;
/** Build search list size (default 100). */
@property (nonatomic) NSInteger diskannL;
/** Pruning parameter α (default 1.2). */
@property (nonatomic) float     diskannAlpha;

/* ── LSH ────────────────────────────────────────────────────────────────────── */

/** Number of hash tables (default 10). */
@property (nonatomic) NSInteger lshL;
/** Hash functions per table (default 8). */
@property (nonatomic) NSInteger lshK;
/** Bucket width for E2LSH (default 10.0). */
@property (nonatomic) float     lshW;

/* ── ScaNN ──────────────────────────────────────────────────────────────────── */

/** Coarse IVF partitions (default 128). */
@property (nonatomic) NSInteger scannNlist;
/** Partitions to probe at search time (default 32). */
@property (nonatomic) NSInteger scannNprobe;
/** PQ sub-spaces (default 8). */
@property (nonatomic) NSInteger scannPqM;
/** Bits per sub-code: 4 or 8 (default 8). */
@property (nonatomic) NSInteger scannPqBits;
/** Candidates to exact-rerank (default 100). */
@property (nonatomic) NSInteger scannRerankK;
/** Anisotropic penalty η (default 0.2). */
@property (nonatomic) float     scannAqEta;

/** Returns a new PSTParams instance initialised to all defaults. */
+ (instancetype)defaultParams NS_SWIFT_NAME(default());

@end

NS_ASSUME_NONNULL_END
